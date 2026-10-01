# CPU paths use column-major loops and preserve the generic GPU dispatch.
native_conv_weights(::Val{:cpu}, weight) = transpose(permutedims(weight))
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

function cpu_mlp_workspace(layers, sequence_length)
    get(ENV, "JEFF_CPU_MLP_WORKSPACE", "0") == "1" || return nothing
    width = size(first(layers).mlp.gate, 2)
    all(layer -> size(layer.mlp.gate, 2) == width, layers) || return nothing
    return (
        full = (
            zeros(Float32, width, sequence_length),
            zeros(Float32, width, sequence_length),
        ),
        final = (zeros(Float32, width, 1), zeros(Float32, width, 1)),
    )
end

native_mlp(mlp, x::Matrix{Float32}, ::Nothing) = native_mlp(mlp, x)
function native_mlp(mlp, x::Matrix{Float32}, buffers::Tuple)
    gate, up = buffers
    mul!(gate, transpose(mlp.gate), x)
    mul!(up, transpose(mlp.up), x)
    return native_linear(mlp.down, cpu_owned_mlp_gate!(gate, up))
end

function cpu_layer_with_mlp_workspace(layer, x, mask, cfg, buffers, delta_buffers = nothing)
    normalized = native_rms(x, layer.input_norm, cfg.eps)
    mixed =
        layer.attention.kind == :full ?
        full_attention(layer.attention, normalized, mask, cfg) :
        delta_attention(layer.attention, normalized, mask, cfg, delta_buffers)
    residual, normalized = native_residual_rms(x, mixed, layer.post_norm, cfg.eps)
    return native_residual_add!(residual, native_mlp(layer.mlp, normalized, buffers))
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

cpu_delta_state_product!(output, state, rhs, alpha, beta) =
    mul!(output, state, rhs, alpha, beta)

function cpu_delta_chunk_size()
    size = parse(Int, get(ENV, "JEFF_CPU_DELTA_CHUNK_SIZE", "64"))
    size > 0 || throw(ArgumentError("Delta chunk size must be positive."))
    return size
end

function cpu_delta_workers(cfg)
    requested =
        parse(Int, get(ENV, "JEFF_CPU_DELTA_WORKERS", string(Threads.nthreads(:default))))
    requested > 0 || throw(ArgumentError("Delta worker count must be positive."))
    return min(requested, Threads.nthreads(:default), cfg.value_heads)
end

function cpu_delta_worker_workspace(cfg, sequence_length)
    requested_chunk_size = cpu_delta_chunk_size()
    chunk_size = get(ENV, "JEFF_CPU_RECURRENT_DELTA", "0") == "1" ? 1 : requested_chunk_size
    full_size = min(chunk_size, sequence_length)
    full = cpu_delta_buffers(cfg, full_size)
    tail_size = mod(sequence_length, chunk_size)
    tail =
        tail_size == 0 || tail_size == full_size ? full : cpu_delta_buffers(cfg, tail_size)
    return (; state = zeros(Float32, cfg.value_dim, cfg.key_dim), full, tail)
end

function cpu_delta_workspace(cfg, sequence_length)
    get(ENV, "JEFF_CPU_DELTA_WORKSPACE", "0") == "1" || return nothing
    workers = get(ENV, "JEFF_CPU_PARALLEL_HEADS", "0") == "1" ? cpu_delta_workers(cfg) : 1
    return [cpu_delta_worker_workspace(cfg, sequence_length) for _ = 1:workers]
end

function delta_attention(attention, x::Matrix{Float32}, mask, cfg, workspace = nothing)
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
    if get(ENV, "JEFF_CPU_PARALLEL_HEADS", "0") == "1" && Threads.nthreads(:default) > 1
        workers = cpu_delta_workers(cfg)
        @sync for worker = 1:workers
            Threads.@spawn cpu_delta_heads!(
                worker:workers:cfg.value_heads,
                out,
                q,
                k,
                v,
                beta,
                decay,
                z,
                attention,
                cfg,
                groups,
                inplace_rms,
                workspace === nothing ? nothing : workspace[worker],
            )
        end
    else
        cpu_delta_heads!(
            1:cfg.value_heads,
            out,
            q,
            k,
            v,
            beta,
            decay,
            z,
            attention,
            cfg,
            groups,
            inplace_rms,
            workspace === nothing ? nothing : workspace[1],
        )
    end
    return native_linear(attention.out, out)
end

function cpu_delta_heads!(
    heads,
    out,
    q,
    k,
    v,
    beta,
    decay,
    z,
    attention,
    cfg,
    groups,
    inplace_rms,
    workspace = nothing,
)
    sequence_length = size(out, 2)
    owned =
        workspace === nothing ? cpu_delta_worker_workspace(cfg, sequence_length) : workspace
    if get(ENV, "JEFF_CPU_RECURRENT_DELTA", "0") == "1"
        return cpu_delta_recurrent_heads!(
            heads,
            out,
            q,
            k,
            v,
            beta,
            decay,
            z,
            attention,
            cfg,
            groups,
            owned,
        )
    end
    state = owned.state
    chunk_size = cpu_delta_chunk_size()
    full_size = min(chunk_size, sequence_length)
    full_buffers = owned.full
    tail_buffers = owned.tail
    for head in heads
        fill!(state, 0.0f0)
        kh = cld(head, groups)
        for start = 1:chunk_size:sequence_length
            span = start:min(start+chunk_size-1, sequence_length)
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
            cpu_delta_state_product!(corrections, state, transpose(keys_rhs), -1.0f0, 1.0f0)
            intra = buffers.intra
            mul!(intra, transpose(kc), qc)
            intra .*= transpose(pair_decay)
            buffers.scaled_query .= qc .* exp_decay
            result = buffers.result
            cpu_delta_state_product!(result, state, buffers.scaled_query, 1.0f0, 0.0f0)
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
    return nothing
end

function cpu_delta_recurrent_heads!(
    heads,
    out,
    q,
    k,
    v,
    beta,
    decay,
    z,
    attention,
    cfg,
    groups,
    owned,
)
    state = owned.state
    # The first scratch columns are worker-owned and no chunk products use them
    # on this path. Reset before each token; reset the state before each head.
    projected = @view owned.full.corrections[:, 1]
    result = @view owned.full.result[:, 1]
    for head in heads
        fill!(state, 0.0f0)
        kh = cld(head, groups)
        for token in axes(out, 2)
            fill!(projected, 0.0f0)
            factor = exp(decay[head, token])
            for column = 1:cfg.key_dim
                key = k[column, kh, token]
                @inbounds @simd for row = 1:cfg.value_dim
                    projected[row] += state[row, column] * key
                end
            end
            @inbounds @simd for row = 1:cfg.value_dim
                projected[row] =
                    beta[head, token] * (v[row, head, token] - factor * projected[row])
            end
            fill!(result, 0.0f0)
            for column = 1:cfg.key_dim
                key = k[column, kh, token]
                query = q[column, kh, token]
                @inbounds @simd for row = 1:cfg.value_dim
                    updated = factor * state[row, column] + projected[row] * key
                    state[row, column] = updated
                    result[row] += updated * query
                end
            end
            scale = inv(sqrt(sum(abs2, result) / cfg.value_dim + cfg.eps))
            offset = (head - 1) * cfg.value_dim
            @inbounds for row = 1:cfg.value_dim
                out[offset+row, token] =
                    ((result[row] * scale) * attention.norm[row]) *
                    native_silu(z[row, head, token])
            end
        end
    end
    return nothing
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
    cpu_convolution!(output, input, weight)
    if Owned
        native_cpu_silu!(output, input)
    else
        native_cpu_silu!(output)
    end
    return output
end

function cpu_convolution!(output, input, weight)
    channels, sequence_length = size(input)
    kernel = size(weight, 1)
    for index = 1:kernel
        lag = kernel - index
        # Safetensors weights are strided reinterpret wrappers. Materialize one
        # tap so the token loop reads a contiguous, SIMD-friendly channel vector.
        coefficients =
            weight isa Transpose{Float32,Matrix{Float32}} ?
            @view(parent(weight)[:, index]) : collect(@view weight[index, :])
        for token = (lag+1):sequence_length
            @inbounds @simd for channel = 1:channels
                output[channel, token] += input[channel, token-lag] * coefficients[channel]
            end
        end
    end
    return output
end

function native_hidden_forward(hidden::Matrix{Float32}, layers, mask, final_norm, cfg)
    isempty(layers) && return native_rms(hidden[:, end:end], final_norm, cfg.eps)
    workspace = cpu_mlp_workspace(layers, size(hidden, 2))
    delta_buffers = cpu_delta_workspace(cfg, size(hidden, 2))
    if workspace === nothing
        if delta_buffers === nothing
            return cpu_hidden_forward(
                hidden,
                layers,
                mask,
                final_norm,
                cfg,
                nothing,
                nothing,
            )
        end
        return cpu_hidden_forward(
            hidden,
            layers,
            mask,
            final_norm,
            cfg,
            nothing,
            delta_buffers,
        )
    end
    if delta_buffers === nothing
        return cpu_hidden_forward(hidden, layers, mask, final_norm, cfg, workspace, nothing)
    end
    return cpu_hidden_forward(
        hidden,
        layers,
        mask,
        final_norm,
        cfg,
        workspace,
        delta_buffers,
    )
end

function cpu_final_full_attention(attention, x, mask, cfg)
    n = size(x, 2)
    qgate = reshape(native_linear(attention.q, x[:, end:end]), 2cfg.head_dim, cfg.heads, 1)
    q = native_rms(qgate[1:cfg.head_dim, :, :], attention.q_norm, cfg.eps)
    # The final query keeps its original position; K/V still cover the context.
    half = cfg.rotary_dim ÷ 2
    for head = 1:cfg.heads, row = 1:half
        frequency = inv(cfg.rope_theta ^ (Float32(2(row - 1)) / Float32(cfg.rotary_dim)))
        angle = frequency * Float32(n - 1)
        cosine, sine = cos(angle), sin(angle)
        a, b = q[row, head, 1], q[half+row, head, 1]
        q[row, head, 1] = a * cosine - b * sine
        q[half+row, head, 1] = b * cosine + a * sine
    end
    k = native_rope(
        native_rms(
            reshape(native_linear(attention.k, x), cfg.head_dim, cfg.kv_heads, n),
            attention.k_norm,
            cfg.eps,
        ),
        cfg,
    )
    v = reshape(native_linear(attention.v, x), cfg.head_dim, cfg.kv_heads, n)
    out = Matrix{Float32}(undef, cfg.head_dim * cfg.heads, 1)
    groups = cfg.heads ÷ cfg.kv_heads
    for head = 1:cfg.heads
        kv = cld(head, groups)
        scores = transpose(@view(k[:, kv, :])) * @view(q[:, head, :])
        scores ./= sqrt(Float32(cfg.head_dim))
        for token = 1:n
            mask[token] == 1 || (scores[token] += -floatmax(Float32))
        end
        probabilities = exp.(scores .- maximum(scores))
        probabilities ./= sum(probabilities)
        values = @view(v[:, kv, :]) * probabilities
        @views out[((head-1)*cfg.head_dim+1):(head*cfg.head_dim), :] .=
            values .* native_sigmoid.(qgate[(cfg.head_dim+1):end, head, :])
    end
    return native_linear(attention.out, out)
end

function cpu_hidden_forward(hidden, layers, mask, final_norm, cfg, workspace, delta_buffers)
    for index = 1:(length(layers)-1)
        hidden = cpu_layer_with_mlp_workspace(
            layers[index],
            hidden,
            mask,
            cfg,
            workspace === nothing ? nothing : workspace.full,
            delta_buffers,
        )
    end
    # Attention consumes the full context. The final MLP is position-wise,
    # and only the last position contributes to Jeff's trained readout.
    layer = last(layers)
    normalized = native_rms(hidden, layer.input_norm, cfg.eps)
    mixed =
        layer.attention.kind == :full ?
        (
            get(ENV, "JEFF_CPU_FINAL_QUERY", "0") == "1" ?
            cpu_final_full_attention(layer.attention, normalized, mask, cfg) :
            full_attention(layer.attention, normalized, mask, cfg)
        ) : delta_attention(layer.attention, normalized, mask, cfg, delta_buffers)
    residual = @views hidden[:, end:end] .+ mixed[:, end:end]
    normalized = native_rms(residual, layer.post_norm, cfg.eps)
    buffers = workspace === nothing ? nothing : workspace.final
    native_residual_add!(residual, native_mlp(layer.mlp, normalized, buffers))
    return native_rms(residual, final_norm, cfg.eps)
end
