# CPU paths use column-major loops and preserve the generic GPU dispatch.
function native_cpu_silu!(output)
    output .= native_silu.(output)
    return output
end
native_cpu_silu!(output, scratch) = native_cpu_silu!(output)

# Both projection outputs are owned by this call and disposable after gating.
cpu_owned_mlp_gate!(gate, up) = native_mlp_gate!(gate, up)

function native_mlp(mlp, x::Matrix{Float32})
    gate = native_linear(mlp.gate, x)
    up = native_linear(mlp.up, x)
    return native_linear(mlp.down, cpu_owned_mlp_gate!(gate, up))
end

function cpu_delta_rms!(output::Matrix{Float32}, weight::AbstractVector{Float32}, eps)
    width = size(output, 1)
    length(weight) == width || throw(DimensionMismatch("RMS weight width differs."))
    for column in axes(output, 2)
        scale = inv(sqrt(sum(abs2, @view(output[:, column])) / width + eps))
        @inbounds for row in axes(output, 1)
            output[row, column] = (output[row, column] * scale) * weight[row]
        end
    end
    return output
end

function cpu_delta_buffers(cfg, n)
    return (
        pair = Matrix{Float32}(undef, n, n),
        weighted = Matrix{Float32}(undef, cfg.key_dim, n),
        system = Matrix{Float32}(undef, n, n),
        values = Matrix{Float32}(undef, n, cfg.value_dim),
        keys = Matrix{Float32}(undef, n, cfg.key_dim),
        corrections = Matrix{Float32}(undef, cfg.value_dim, n),
        intra = Matrix{Float32}(undef, n, n),
        scaled_query = Matrix{Float32}(undef, cfg.key_dim, n),
        result = Matrix{Float32}(undef, cfg.value_dim, n),
        ending_keys = Matrix{Float32}(undef, cfg.key_dim, n),
    )
end

function delta_attention(attention, x::Matrix{Float32}, mask, cfg)
    sequence_length = size(x, 2)
    masked = x .* reshape(Float32.(mask), 1, :)
    mixed = cpu_owned_causal_depthwise(native_linear(attention.qkv, masked), attention.conv)
    key_width = cfg.key_dim * cfg.key_heads
    q = reshape(
        copy(@view mixed[1:key_width, :]),
        cfg.key_dim,
        cfg.key_heads,
        sequence_length,
    )
    k = reshape(
        copy(@view mixed[(key_width+1):2key_width, :]),
        cfg.key_dim,
        cfg.key_heads,
        sequence_length,
    )
    v = reshape(
        @view(mixed[(2key_width+1):end, :]),
        cfg.value_dim,
        cfg.value_heads,
        sequence_length,
    )
    # Q/K heads are shared by multiple value heads. Normalize once, not once
    # per value head, and let chunk views reference the normalized storage.
    q ./= sqrt.(sum(abs2, q; dims = 1) .+ 1.0f-6) .* sqrt(Float32(cfg.key_dim))
    k ./= sqrt.(sum(abs2, k; dims = 1) .+ 1.0f-6)
    z = reshape(
        native_linear(attention.z, masked),
        cfg.value_dim,
        cfg.value_heads,
        sequence_length,
    )
    beta = native_sigmoid.(native_linear(attention.b, masked))
    decay =
        attention.a_decay .*
        native_softplus.(native_linear(attention.a, masked) .+ attention.dt_bias)
    out = Matrix{Float32}(undef, cfg.value_dim * cfg.value_heads, sequence_length)
    groups = cfg.value_heads ÷ cfg.key_heads
    inplace_rms = get(ENV, "JEFF_CPU_INPLACE_DELTA_RMS", "0") == "1"
    state = zeros(Float32, cfg.value_dim, cfg.key_dim)
    full_size = min(64, sequence_length)
    full_buffers = cpu_delta_buffers(cfg, full_size)
    tail_size = mod(sequence_length, 64)
    tail_buffers =
        tail_size == 0 || tail_size == full_size ? full_buffers :
        cpu_delta_buffers(cfg, tail_size)
    for head = 1:cfg.value_heads
        fill!(state, 0.0f0)
        kh = cld(head, groups)
        for start = 1:64:sequence_length
            span = start:min(start+63, sequence_length)
            n = length(span)
            buffers = n == full_size ? full_buffers : tail_buffers
            qc = @view q[:, kh, span]
            kc = @view k[:, kh, span]
            vc = @view v[:, head, span]
            bc = @view beta[head:head, span]
            cumulative = cumsum(@view(decay[head:head, span]); dims = 2)
            pair_decay = buffers.pair
            for j = 1:n, i = 1:n
                pair_decay[i, j] = i >= j ? exp(cumulative[i] - cumulative[j]) : 0.0f0
            end
            weighted_keys = buffers.weighted
            weighted_keys .= kc .* bc
            system = buffers.system
            mul!(system, transpose(weighted_keys), kc)
            system .*= pair_decay
            exp_decay = exp.(cumulative)
            values_rhs = buffers.values
            values_rhs .= transpose(vc) .* transpose(bc)
            keys_rhs = buffers.keys
            keys_rhs .= transpose(weighted_keys) .* transpose(exp_decay)
            # RHS buffers belong to this chunk and can be overwritten directly.
            BLAS.trsm!('L', 'L', 'N', 'U', 1.0f0, system, values_rhs)
            BLAS.trsm!('L', 'L', 'N', 'U', 1.0f0, system, keys_rhs)
            corrections = buffers.corrections
            corrections .= transpose(values_rhs)
            mul!(corrections, state, transpose(keys_rhs), -1.0f0, 1.0f0)
            intra = buffers.intra
            mul!(intra, transpose(kc), qc)
            intra .*= transpose(pair_decay)
            buffers.scaled_query .= qc .* exp_decay
            result = buffers.result
            mul!(result, state, buffers.scaled_query)
            mul!(result, corrections, intra, 1.0f0, 1.0f0)
            ending_keys = buffers.ending_keys
            ending_keys .= kc .* exp.(cumulative[end] .- cumulative)
            mul!(state, corrections, transpose(ending_keys), 1.0f0, exp(cumulative[end]))
            normalized =
                inplace_rms ? cpu_delta_rms!(result, attention.norm, cfg.eps) :
                native_rms(result, attention.norm, cfg.eps; centered = false)
            destination = @view out[((head-1)*cfg.value_dim+1):(head*cfg.value_dim), span]
            destination .= normalized .* native_silu.(@view z[:, head, span])
        end
    end
    return native_linear(attention.out, out)
end

causal_depthwise(input::Matrix{Float32}, weight::AbstractMatrix{Float32}) =
    cpu_depthwise(input, weight, Val(false))
cpu_owned_causal_depthwise(input::Matrix{Float32}, weight::AbstractMatrix{Float32}) =
    cpu_depthwise(input, weight, Val(true))

function cpu_depthwise(
    input::Matrix{Float32},
    weight::AbstractMatrix{Float32},
    ::Val{Owned},
) where {Owned}
    kernel = size(weight, 1)
    channels, sequence_length = size(input)
    size(weight, 2) == channels ||
        throw(DimensionMismatch("Convolution channel counts differ."))
    output = zeros(Float32, size(input))
    for index = 1:kernel
        lag = kernel - index
        for token = (lag+1):sequence_length
            @inbounds for channel = 1:channels
                output[channel, token] += input[channel, token-lag] * weight[index, channel]
            end
        end
    end
    if Owned
        native_cpu_silu!(output, input)
    else
        native_cpu_silu!(output)
    end
    return output
end

function native_hidden_forward(hidden::Matrix{Float32}, layers, mask, final_norm, cfg)
    isempty(layers) && return native_rms(hidden[:, end:end], final_norm, cfg.eps)
    for index = 1:(length(layers)-1)
        hidden = native_layer(layers[index], hidden, mask, cfg)
    end
    # Attention consumes the full context. The final MLP is position-wise,
    # and only the last position contributes to Jeff's trained readout.
    layer = last(layers)
    normalized = native_rms(hidden, layer.input_norm, cfg.eps)
    mixed =
        layer.attention.kind == :full ?
        full_attention(layer.attention, normalized, mask, cfg) :
        delta_attention(layer.attention, normalized, mask, cfg)
    residual = @views hidden[:, end:end] .+ mixed[:, end:end]
    normalized = native_rms(residual, layer.post_norm, cfg.eps)
    native_residual_add!(residual, native_mlp(layer.mlp, normalized))
    return native_rms(residual, final_norm, cfg.eps)
end
