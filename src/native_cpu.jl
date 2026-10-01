# CPU paths use column-major loops and preserve the generic GPU dispatch.
native_conv_weights(::Val{:cpu}, weight) = transpose(permutedims(weight))
cpu_portable_silu!(output) = false
cpu_portable_gate!(gate, up) = false
cpu_portable_softmax!(scores) = false
function native_cpu_silu!(output)
    cpu_portable_silu!(output) && return output
    output .= native_silu.(output)
    return output
end
native_cpu_silu!(output, scratch) = native_cpu_silu!(output)

# Both projection outputs are owned by this call and disposable after gating.
cpu_owned_mlp_gate!(gate, up) =
    cpu_portable_gate!(gate, up) ? gate : native_mlp_gate!(gate, up)

function cpu_projection_block!(output, weight, input, rows, beta = 0.0f0)
    lhs =
        weight isa Transpose{Float32,Matrix{Float32}} ? @view(parent(weight)[rows, :]) :
        transpose(@view(weight[:, rows]))
    if iszero(beta)
        mul!(@view(output[rows, :]), lhs, input)
    else
        mul!(@view(output[rows, :]), lhs, input, 1.0f0, beta)
    end
    return nothing
end

function cpu_projection_blas_threads()
    scoped = get(task_local_storage(), :jeff_cpu_projection_blas_threads, nothing)
    return scoped isa Int ? scoped : BLAS.get_num_threads()
end

function cpu_projection_scope(f::F) where {F}
    if !cpu_setting(:parallel_projections) || !cpu_setting(:projection_thread_scope)
        return f()
    end
    # Immutable policy owned by this task/forward, restored by Base even on
    # exceptions. BLAS configuration must not change during a running forward.
    return task_local_storage(f, :jeff_cpu_projection_blas_threads, BLAS.get_num_threads())
end

function cpu_full_heads!(heads, out, q, k, v, qgate, scores_mask, cfg)
    groups = cfg.heads ÷ cfg.kv_heads
    for head in heads
        kv = cld(head, groups)
        scores =
            native_matmul(transpose(@view(k[:, kv, :])), @view(q[:, head, :])) ./
            sqrt(Float32(cfg.head_dim)) .+ scores_mask
        if !cpu_portable_softmax!(scores)
            scores .= exp.(scores .- maximum(scores; dims = 1))
            scores ./= sum(scores; dims = 1)
        end
        values = native_matmul(@view(v[:, kv, :]), scores)
        gate = native_sigmoid.(@view(qgate[(cfg.head_dim+1):end, head, :]))
        @views out[((head-1)*cfg.head_dim+1):(head*cfg.head_dim), :] .= values .* gate
    end
    return nothing
end

function native_full_heads!(out::Matrix{Float32}, q, k, v, qgate, scores_mask, cfg)
    workers = min(Threads.nthreads(:default), cfg.heads)
    if !cpu_setting(:parallel_full_heads)
        return invoke(
            native_full_heads!,
            Tuple{Any,Any,Any,Any,Any,Any,Any},
            out,
            q,
            k,
            v,
            qgate,
            scores_mask,
            cfg,
        )
    end
    if workers <= 1 || size(out, 2) < 16 || cpu_projection_blas_threads() != 1
        cpu_full_heads!(1:cfg.heads, out, q, k, v, qgate, scores_mask, cfg)
        return out
    end
    settings = cpu_settings()
    @sync for worker = 1:workers
        Threads.@spawn with_cpu_settings(settings) do
            cpu_full_heads!(worker:workers:cfg.heads, out, q, k, v, qgate, scores_mask, cfg)
        end
    end
    return out
end

function cpu_full_workspace(cfg, n)
    half = cfg.rotary_dim ÷ 2
    frequencies = inv.(
        cfg.rope_theta .^ (Float32.(0:2:(cfg.rotary_dim-1)) ./ Float32(cfg.rotary_dim)),
    )
    angles = frequencies .* permutedims(Float32.(0:(n-1)))
    return (
        qgate = zeros(Float32, 2cfg.head_dim * cfg.heads, n),
        q = zeros(Float32, cfg.head_dim, cfg.heads, n),
        k = zeros(Float32, cfg.head_dim * cfg.kv_heads, n),
        v = zeros(Float32, cfg.head_dim * cfg.kv_heads, n),
        out = zeros(Float32, cfg.head_dim * cfg.heads, n),
        projected = zeros(Float32, cfg.hidden, n),
        mask = zeros(Float32, n, n),
        cosines = reshape(cos.(angles), half, n),
        sines = reshape(sin.(angles), half, n),
        heads = [
            (scores = zeros(Float32, n, n), values = zeros(Float32, cfg.head_dim, n)) for
            _ = 1:min(Threads.nthreads(:default), cfg.heads)
        ],
    )
end

function cpu_full_rms!(output, input, weight, eps)
    width = size(output, 1)
    for token in axes(output, 3), head in axes(output, 2)
        scale = inv(sqrt(sum(abs2, @view(input[:, head, token])) / width + eps))
        @inbounds @simd for row in axes(output, 1)
            output[row, head, token] =
                (input[row, head, token] * scale) * (1.0f0 + weight[row])
        end
    end
    return output
end

function cpu_full_rope!(output, cosines, sines)
    half = size(cosines, 1)
    for token in axes(output, 3), head in axes(output, 2)
        @inbounds @simd for row = 1:half
            a, b = output[row, head, token], output[half+row, head, token]
            cosine, sine = cosines[row, token], sines[row, token]
            output[row, head, token] = a * cosine - b * sine
            output[half+row, head, token] = b * cosine + a * sine
        end
    end
    return output
end

function cpu_full_workspace_heads!(heads, q, k, v, qgate, cfg, buffers, scratch)
    groups = cfg.heads ÷ cfg.kv_heads
    for head in heads
        kv = cld(head, groups)
        scores, values = scratch
        mul!(scores, transpose(@view(k[:, kv, :])), @view(q[:, head, :]))
        scores .= scores ./ sqrt(Float32(cfg.head_dim)) .+ buffers.mask
        if !cpu_portable_softmax!(scores)
            scores .= exp.(scores .- maximum(scores; dims = 1))
            scores ./= sum(scores; dims = 1)
        end
        mul!(values, @view(v[:, kv, :]), scores)
        @views buffers.out[((head-1)*cfg.head_dim+1):(head*cfg.head_dim), :] .=
            values .* native_sigmoid.(qgate[(cfg.head_dim+1):end, head, :])
    end
    return nothing
end

cpu_full_attention(attention, x, mask, cfg, ::Nothing) =
    full_attention(attention, x, mask, cfg)
function cpu_full_attention(attention, x, mask, cfg, buffers)
    n = size(x, 2)
    cpu_projection!(buffers.qgate, attention.q, x)
    qgate = reshape(buffers.qgate, 2cfg.head_dim, cfg.heads, n)
    cpu_full_rms!(buffers.q, @view(qgate[1:cfg.head_dim, :, :]), attention.q_norm, cfg.eps)
    cpu_full_rope!(buffers.q, buffers.cosines, buffers.sines)
    cpu_projection!(buffers.k, attention.k, x)
    k = reshape(buffers.k, cfg.head_dim, cfg.kv_heads, n)
    cpu_full_rms!(k, k, attention.k_norm, cfg.eps)
    cpu_full_rope!(k, buffers.cosines, buffers.sines)
    cpu_projection!(buffers.v, attention.v, x)
    v = reshape(buffers.v, cfg.head_dim, cfg.kv_heads, n)
    for column = 1:n, row = 1:n
        buffers.mask[row, column] =
            row <= column && mask[row] == 1 ? 0.0f0 : -floatmax(Float32)
    end
    workers =
        cpu_setting(:parallel_full_heads) && n >= 16 && cpu_projection_blas_threads() == 1 ?
        min(Threads.nthreads(:default), cfg.heads) : 1
    if workers == 1
        cpu_full_workspace_heads!(
            1:cfg.heads,
            buffers.q,
            k,
            v,
            qgate,
            cfg,
            buffers,
            first(buffers.heads),
        )
    else
        settings = cpu_settings()
        @sync for worker = 1:workers
            Threads.@spawn with_cpu_settings(settings) do
                cpu_full_workspace_heads!(
                    worker:workers:cfg.heads,
                    buffers.q,
                    k,
                    v,
                    qgate,
                    cfg,
                    buffers,
                    buffers.heads[worker],
                )
            end
        end
    end
    return cpu_projection!(buffers.projected, attention.out, buffers.out)
end

function cpu_projection!(output, weight, input, beta = 0.0f0)
    size(input, 1) == size(weight, 1) &&
    size(output) == (size(weight, 2), size(input, 2)) ||
        throw(DimensionMismatch("Projection shapes differ."))
    if Base.mightalias(output, input) || Base.mightalias(output, weight)
        product = native_matmul(transpose(weight), input)
        if iszero(beta)
            return copyto!(output, product)
        end
        output .= product .+ beta .* output
        return output
    end
    workers = min(Threads.nthreads(:default), size(output, 1))
    if !cpu_setting(:parallel_projections) ||
       workers == 1 ||
       size(output, 1) < 256 ||
       length(output) * size(input, 1) < 1_000_000 ||
       cpu_projection_blas_threads() != 1
        return iszero(beta) ? mul!(output, transpose(weight), input) :
               mul!(output, transpose(weight), input, 1.0f0, beta)
    end
    @sync for worker = 1:workers
        rows =
            (fld((worker-1)*size(output, 1), workers)+1):fld(
                worker*size(output, 1),
                workers,
            )
        Threads.@spawn cpu_projection_block!(output, weight, input, rows, beta)
    end
    return output
end

function native_linear(weight::AbstractMatrix{Float32}, x::Matrix{Float32})
    cpu_setting(:parallel_projections) || return native_matmul(transpose(weight), x)
    output = Matrix{Float32}(undef, size(weight, 2), size(x, 2))
    return cpu_projection!(output, weight, x)
end

function native_mlp(mlp, x::Matrix{Float32})
    gate = native_linear(mlp.gate, x)
    up = native_linear(mlp.up, x)
    return native_linear(mlp.down, cpu_owned_mlp_gate!(gate, up))
end

function cpu_mlp_workspace(layers, sequence_length)
    cpu_setting(:mlp_workspace) || return nothing
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
    cpu_projection!(gate, mlp.gate, x)
    cpu_projection!(up, mlp.up, x)
    return native_linear(mlp.down, cpu_owned_mlp_gate!(gate, up))
end

function cpu_mlp_add!(residual, mlp, x, buffers)
    cpu_setting(:mlp_residual_fusion) ||
        return native_residual_add!(residual, native_mlp(mlp, x, buffers))
    # The residual belongs to this layer. Gate/up remain private workspace;
    # beta=1 accumulates the down projection without a temporary output.
    gate, up = if buffers === nothing
        (native_linear(mlp.gate, x), native_linear(mlp.up, x))
    else
        gate, up = buffers
        cpu_projection!(gate, mlp.gate, x)
        cpu_projection!(up, mlp.up, x)
        (gate, up)
    end
    return cpu_projection!(residual, mlp.down, cpu_owned_mlp_gate!(gate, up), 1.0f0)
end

function cpu_layer_with_mlp_workspace(
    layer,
    x,
    mask,
    cfg,
    buffers,
    delta_buffers = nothing,
    projections = nothing,
    normalization = nothing,
)
    normalized =
        normalization === nothing ? native_rms(x, layer.input_norm, cfg.eps) :
        cpu_rms!(normalization.normalized, x, layer.input_norm, cfg.eps)
    mixed =
        layer.attention.kind == :full ?
        cpu_full_attention(
            layer.attention,
            normalized,
            mask,
            cfg,
            normalization === nothing ? nothing : normalization.attention,
        ) :
        delta_attention(layer.attention, normalized, mask, cfg, delta_buffers, projections)
    residual, normalized = if normalization === nothing
        native_residual_rms(x, mixed, layer.post_norm, cfg.eps)
    else
        # Both matrices belong to this forward. After attention completes,
        # the pre-attention normalization can be overwritten for the MLP.
        normalization.residual .= x .+ mixed
        normalization.residual,
        cpu_rms!(
            normalization.normalized,
            normalization.residual,
            layer.post_norm,
            cfg.eps,
        )
    end
    return cpu_mlp_add!(residual, layer.mlp, normalized, buffers)
end

function cpu_rms!(
    output::Matrix{Float32},
    input::Matrix{Float32},
    weight::AbstractVector{Float32},
    eps,
)
    size(output) == size(input) && length(weight) == size(input, 1) ||
        throw(DimensionMismatch("RMS workspace shapes differ."))
    Base.mightalias(output, weight) && (weight = copy(weight))
    width = size(input, 1)
    for column in axes(input, 2)
        scale = inv(sqrt(sum(abs2, @view(input[:, column])) / width + eps))
        @inbounds @simd for row in axes(input, 1)
            output[row, column] = (input[row, column] * scale) * (1.0f0 + weight[row])
        end
    end
    return output
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
    size = cpu_setting(:delta_chunk_size)
    size > 0 || throw(ArgumentError("Delta chunk size must be positive."))
    return size
end

function cpu_delta_workers(cfg)
    requested = cpu_setting(:delta_workers)
    requested > 0 || throw(ArgumentError("Delta worker count must be positive."))
    return min(requested, Threads.nthreads(:default), cfg.value_heads)
end

function cpu_delta_worker_workspace(cfg, sequence_length)
    requested_chunk_size = cpu_delta_chunk_size()
    chunk_size = cpu_setting(:recurrent_delta) ? 1 : requested_chunk_size
    full_size = min(chunk_size, sequence_length)
    full = cpu_delta_buffers(cfg, full_size)
    tail_size = mod(sequence_length, chunk_size)
    tail =
        tail_size == 0 || tail_size == full_size ? full : cpu_delta_buffers(cfg, tail_size)
    return (; state = zeros(Float32, cfg.value_dim, cfg.key_dim), full, tail)
end

function cpu_delta_workspace(cfg, sequence_length)
    cpu_setting(:delta_workspace) || return nothing
    workers = cpu_setting(:parallel_heads) ? cpu_delta_workers(cfg) : 1
    return [cpu_delta_worker_workspace(cfg, sequence_length) for _ = 1:workers]
end

function cpu_normalize_delta_heads!(values::Array{Float32,3}, multiplier::Float32)
    if !cpu_setting(:delta_norm_loop)
        values ./= sqrt.(sum(abs2, values; dims = 1) .+ 1.0f-6) .* multiplier
        return values
    end
    for token in axes(values, 3), head in axes(values, 2)
        squared = 0.0f0
        @inbounds @simd for row in axes(values, 1)
            squared += abs2(values[row, head, token])
        end
        denominator = sqrt(squared + 1.0f-6) * multiplier
        @inbounds @simd for row in axes(values, 1)
            values[row, head, token] /= denominator
        end
    end
    return values
end

function cpu_delta_projection_workspace(cfg, n)
    cpu_setting(:delta_projection_workspace) || return nothing
    key_width = cfg.key_dim * cfg.key_heads
    value_width = cfg.value_dim * cfg.value_heads
    return (
        masked = zeros(Float32, cfg.hidden, n),
        qkv = zeros(Float32, 2key_width+value_width, n),
        mixed = zeros(Float32, 2key_width+value_width, n),
        q = zeros(Float32, cfg.key_dim, n, cfg.key_heads),
        k = zeros(Float32, cfg.key_dim, n, cfg.key_heads),
        v = zeros(Float32, cfg.value_dim, n, cfg.value_heads),
        z = zeros(Float32, value_width, n),
        gate = zeros(Float32, cfg.value_dim, n, cfg.value_heads),
        beta = zeros(Float32, cfg.value_heads, n),
        decay = zeros(Float32, cfg.value_heads, n),
        out = zeros(Float32, value_width, n),
        projected = zeros(Float32, cfg.hidden, n),
    )
end

function cpu_pack_delta_heads!(packed::Array{Float32,3}, input::Matrix{Float32}, offset = 0)
    width, n, heads = size(packed)
    size(input, 2) == n && 0 <= offset && offset + width * heads <= size(input, 1) ||
        throw(DimensionMismatch("Packed Delta head shapes differ."))
    Base.mightalias(packed, input) && (input = copy(input))
    # A head's entire sequence is contiguous, unlike a token-major projection.
    # Chunk GEMMs can read consecutive columns without crossing other heads.
    for head = 1:heads, token = 1:n
        start = offset + (head - 1) * width
        @inbounds @simd for row = 1:width
            packed[row, token, head] = input[start+row, token]
        end
    end
    return packed
end

function cpu_delta_prepare(attention, x, mask, cfg, ::Nothing)
    sequence_length = size(x, 2)
    masked = x .* reshape(Float32.(mask), 1, :)
    mixed = cpu_owned_causal_depthwise(native_linear(attention.qkv, masked), attention.conv)
    key_width = cfg.key_dim * cfg.key_heads
    q = cpu_pack_delta_heads!(
        Array{Float32}(undef, cfg.key_dim, sequence_length, cfg.key_heads),
        mixed,
    )
    k = cpu_pack_delta_heads!(
        Array{Float32}(undef, cfg.key_dim, sequence_length, cfg.key_heads),
        mixed,
        key_width,
    )
    v = cpu_pack_delta_heads!(
        Array{Float32}(undef, cfg.value_dim, sequence_length, cfg.value_heads),
        mixed,
        2key_width,
    )
    # Q/K heads are shared by multiple value heads. Normalize once, not once
    # per value head, and let chunk views reference the normalized storage.
    cpu_normalize_delta_heads!(q, sqrt(Float32(cfg.key_dim)))
    cpu_normalize_delta_heads!(k, 1.0f0)
    # Activate the owned projection once, so head loops only multiply the gate.
    z = cpu_pack_delta_heads!(
        Array{Float32}(undef, cfg.value_dim, sequence_length, cfg.value_heads),
        native_cpu_silu!(native_linear(attention.z, masked)),
    )
    beta = native_sigmoid.(native_linear(attention.b, masked))
    decay =
        attention.a_decay .*
        native_softplus.(native_linear(attention.a, masked) .+ attention.dt_bias)
    out = Matrix{Float32}(undef, cfg.value_dim * cfg.value_heads, sequence_length)
    return q, k, v, z, beta, decay, out
end

function cpu_delta_prepare(attention, x, mask, cfg, buffers::NamedTuple)
    size(buffers.masked) == size(x) ||
        throw(DimensionMismatch("Delta projection workspace shape differs."))
    size(attention.conv, 2) == size(buffers.qkv, 1) ||
        throw(DimensionMismatch("Convolution channels differ."))
    buffers.masked .= x .* reshape(Float32.(mask), 1, :)
    cpu_projection!(buffers.qkv, attention.qkv, buffers.masked)
    fill!(buffers.mixed, 0.0f0)
    cpu_convolution!(buffers.mixed, buffers.qkv, attention.conv)
    native_cpu_silu!(buffers.mixed, buffers.qkv)
    n = size(x, 2)
    width = cfg.key_dim * cfg.key_heads
    cpu_pack_delta_heads!(buffers.q, buffers.mixed)
    cpu_pack_delta_heads!(buffers.k, buffers.mixed, width)
    cpu_pack_delta_heads!(buffers.v, buffers.mixed, 2width)
    cpu_normalize_delta_heads!(buffers.q, sqrt(Float32(cfg.key_dim)))
    cpu_normalize_delta_heads!(buffers.k, 1.0f0)
    cpu_projection!(buffers.z, attention.z, buffers.masked)
    native_cpu_silu!(buffers.z)
    cpu_pack_delta_heads!(buffers.gate, buffers.z)
    cpu_projection!(buffers.beta, attention.b, buffers.masked)
    buffers.beta .= native_sigmoid.(buffers.beta)
    cpu_projection!(buffers.decay, attention.a, buffers.masked)
    buffers.decay .=
        attention.a_decay .* native_softplus.(buffers.decay .+ attention.dt_bias)
    return buffers.q,
    buffers.k,
    buffers.v,
    buffers.gate,
    buffers.beta,
    buffers.decay,
    buffers.out
end

function delta_attention(
    attention,
    x::Matrix{Float32},
    mask,
    cfg,
    workspace = nothing,
    projections = nothing,
)
    q, k, v, z, beta, decay, out = cpu_delta_prepare(attention, x, mask, cfg, projections)
    groups = cfg.value_heads ÷ cfg.key_heads
    inplace_rms = cpu_setting(:inplace_delta_rms)
    if cpu_setting(:parallel_heads) && Threads.nthreads(:default) > 1
        workers = cpu_delta_workers(cfg)
        settings = cpu_settings()
        @sync for worker = 1:workers
            Threads.@spawn with_cpu_settings(settings) do
                cpu_delta_heads!(
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
    return projections === nothing ? native_linear(attention.out, out) :
           cpu_projection!(projections.projected, attention.out, out)
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
    if cpu_setting(:recurrent_delta)
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
            qc = @view q[:, span, kh]
            kc = @view k[:, span, kh]
            vc = @view v[:, span, head]
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
            destination .= normalized .* @view(z[:, span, head])
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
                key = k[column, token, kh]
                @inbounds @simd for row = 1:cfg.value_dim
                    projected[row] += state[row, column] * key
                end
            end
            @inbounds @simd for row = 1:cfg.value_dim
                projected[row] =
                    beta[head, token] * (v[row, token, head] - factor * projected[row])
            end
            fill!(result, 0.0f0)
            for column = 1:cfg.key_dim
                key = k[column, token, kh]
                query = q[column, token, kh]
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
                    ((result[row] * scale) * attention.norm[row]) * z[row, token, head]
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
    return cpu_projection_scope() do
        cpu_hidden_forward_scoped(hidden, layers, mask, final_norm, cfg)
    end
end

function cpu_hidden_forward_scoped(hidden, layers, mask, final_norm, cfg)
    if isempty(layers)
        return cpu_setting(:final_token_only) ?
               native_rms(hidden[:, end:end], final_norm, cfg.eps) :
               native_rms(hidden, final_norm, cfg.eps)[:, end:end]
    end
    workspace = cpu_mlp_workspace(layers, size(hidden, 2))
    delta_buffers = cpu_delta_workspace(cfg, size(hidden, 2))
    projections = cpu_delta_projection_workspace(cfg, size(hidden, 2))
    normalization =
        cpu_setting(:mlp_workspace) && cpu_setting(:portable_vector_math) ?
        (;
            normalized = similar(hidden),
            residual = similar(hidden),
            attention = cpu_full_workspace(
                cfg,
                any(layer -> layer.attention.kind == :full, layers) ? size(hidden, 2) : 0,
            ),
        ) : nothing
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
                projections,
                normalization,
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
            projections,
            normalization,
        )
    end
    if delta_buffers === nothing
        return cpu_hidden_forward(
            hidden,
            layers,
            mask,
            final_norm,
            cfg,
            workspace,
            nothing,
            projections,
            normalization,
        )
    end
    return cpu_hidden_forward(
        hidden,
        layers,
        mask,
        final_norm,
        cfg,
        workspace,
        delta_buffers,
        projections,
        normalization,
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

function cpu_hidden_forward(
    hidden,
    layers,
    mask,
    final_norm,
    cfg,
    workspace,
    delta_buffers,
    projections = nothing,
    normalization = nothing,
)
    if !cpu_setting(:final_token_only)
        # Match a backbone that computes every position through the final MLP
        # and final normalization, then selects the last token for readout.
        for layer in layers
            hidden = cpu_layer_with_mlp_workspace(
                layer,
                hidden,
                mask,
                cfg,
                workspace === nothing ? nothing : workspace.full,
                delta_buffers,
                projections,
                normalization,
            )
        end
        normalized =
            normalization === nothing ? native_rms(hidden, final_norm, cfg.eps) :
            cpu_rms!(normalization.normalized, hidden, final_norm, cfg.eps)
        return normalized[:, end:end]
    end
    for index = 1:(length(layers)-1)
        hidden = cpu_layer_with_mlp_workspace(
            layers[index],
            hidden,
            mask,
            cfg,
            workspace === nothing ? nothing : workspace.full,
            delta_buffers,
            projections,
            normalization,
        )
    end
    # Attention consumes the full context. The final MLP is position-wise,
    # and only the last position contributes to Jeff's trained readout.
    layer = last(layers)
    normalized =
        normalization === nothing ? native_rms(hidden, layer.input_norm, cfg.eps) :
        cpu_rms!(normalization.normalized, hidden, layer.input_norm, cfg.eps)
    mixed =
        layer.attention.kind == :full ?
        (
            cpu_setting(:final_query) ?
            cpu_final_full_attention(layer.attention, normalized, mask, cfg) :
            cpu_full_attention(
                layer.attention,
                normalized,
                mask,
                cfg,
                normalization === nothing ? nothing : normalization.attention,
            )
        ) :
        delta_attention(layer.attention, normalized, mask, cfg, delta_buffers, projections)
    residual = @views hidden[:, end:end] .+ mixed[:, end:end]
    normalized = native_rms(residual, layer.post_norm, cfg.eps)
    buffers = workspace === nothing ? nothing : workspace.final
    cpu_mlp_add!(residual, layer.mlp, normalized, buffers)
    return native_rms(residual, final_norm, cfg.eps)
end
