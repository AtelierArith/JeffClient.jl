@inline function warp_sum(value::Float32)
    for offset in (16, 8, 4, 2, 1)
        value += CUDA.shfl_xor_sync(CUDA.FULL_MASK, value, offset)
    end
    return value
end

function qk_normalize_kernel!(
    query,
    key,
    mixed,
    dim,
    heads,
    width,
    tokens,
    ::Val{PARTS},
) where {PARTS}
    lane = Int32(CUDA.threadIdx().x)
    group = Int32(CUDA.blockIdx().x) - Int32(1)
    head = rem(group, heads) + Int32(1)
    token = group ÷ heads + Int32(1)
    qs = ntuple(Val(PARTS)) do p
        channel = lane + Int32(32 * (p - 1))
        channel <= dim ? mixed[(head-Int32(1))*dim+channel, token] : 0.0f0
    end
    ks = ntuple(Val(PARTS)) do p
        channel = lane + Int32(32 * (p - 1))
        channel <= dim ? mixed[width+(head-Int32(1))*dim+channel, token] : 0.0f0
    end
    qscale = inv(sqrt(warp_sum(sum(map(abs2, qs))) + 1.0f-6) * sqrt(Float32(dim)))
    kscale = inv(sqrt(warp_sum(sum(map(abs2, ks))) + 1.0f-6))
    for p = 1:PARTS
        channel = lane + Int32(32 * (p - 1))
        if channel <= dim
            query[channel, head, token] = qs[p] * qscale
            key[channel, head, token] = ks[p] * kscale
        end
    end
    return
end

# Each warp owns one value row; each lane owns PARTS key components of
# its recurrent state. No heap arrays or global state buffers inside the scan.
function delta_recurrent_kernel!(
    output,
    query,
    key,
    mixed,
    beta,
    decay,
    key_dim,
    value_dim,
    value_start,
    groups,
    tokens,
    ::Val{PARTS},
) where {PARTS}
    lane = Int32(CUDA.threadIdx().x)
    row =
        (Int32(CUDA.blockIdx().x) - Int32(1))*Int32(CUDA.blockDim().y) +
        Int32(CUDA.threadIdx().y)
    head = Int32(CUDA.blockIdx().y)
    key_head = cld(head, groups)
    state = ntuple(_ -> 0.0f0, Val(PARTS))
    @inbounds for token = Int32(1):tokens
        keys = ntuple(Val(PARTS)) do p
            channel = lane + Int32(32 * (p - 1))
            channel <= key_dim ? key[channel, key_head, token] : 0.0f0
        end
        queries = ntuple(Val(PARTS)) do p
            channel = lane + Int32(32 * (p - 1))
            channel <= key_dim ? query[channel, key_head, token] : 0.0f0
        end
        factor = exp(decay[head, token])
        state = map(s -> s * factor, state)
        prediction = warp_sum(sum(map(*, state, keys)))
        value =
            row <= value_dim ? mixed[value_start+(head-Int32(1))*value_dim+row, token] :
            0.0f0
        correction = (value - prediction) * beta[head, token]
        state = map((s, k) -> s + correction*k, state, keys)
        result = warp_sum(sum(map(*, state, queries)))
        if lane == 1 && row <= value_dim
            output[row, head, token] = result
        end
    end
    return
end

function causal_depthwise_kernel!(output, input, weight, channels, tokens, taps)
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x) +
        Int32(CUDA.threadIdx().x)
    if index <= channels*tokens
        channel = rem(index-Int32(1), channels)+Int32(1)
        token = (index-Int32(1)) ÷ channels + Int32(1)
        value = 0.0f0
        @inbounds for tap = Int32(1):taps
            source = token - (taps-tap)
            if source >= 1
                value += input[channel, source] * weight[tap, channel]
            end
        end
        output[channel, token] = JeffClient.native_silu(value)
    end
    return
end


function JeffClient.causal_depthwise(input::CUDA.StridedCuMatrix{Float32}, weight)
    output = scratch(Float32, size(input))
    CUDA.@cuda threads=256 blocks=cld(length(input), 256) causal_depthwise_kernel!(
        output,
        input,
        weight,
        Int32(size(input, 1)),
        Int32(size(input, 2)),
        Int32(size(weight, 1)),
    )
    return output
end

function JeffClient.delta_attention(attention, x::CUDA.CuMatrix{Float32}, mask, cfg)
    cfg.key_dim <= 256 || return invoke(
        JeffClient.delta_attention,
        Tuple{Any,Any,Any,Any},
        attention,
        x,
        mask,
        cfg,
    )
    tokens = size(x, 2)
    device_mask = mask isa CUDA.CuArray ? mask : JeffClient.native_prepare_mask(x, mask)
    masked = scratch(Float32, size(x))
    masked .= x .* reshape(device_mask, 1, :)
    projected = JeffClient.native_linear(attention.packed_projection, masked)
    qkv_width = size(attention.qkv, 2)
    z_width = size(attention.z, 2)
    a_width = size(attention.a, 2)
    mixed = JeffClient.causal_depthwise(view(projected, 1:qkv_width, :), attention.conv)
    query = scratch(Float32, (cfg.key_dim, cfg.key_heads, tokens))
    key = scratch(Float32, size(query))
    parts = Val(cld(cfg.key_dim, 32))
    width = cfg.key_dim*cfg.key_heads
    CUDA.@cuda threads=32 blocks=cfg.key_heads*tokens qk_normalize_kernel!(
        query,
        key,
        mixed,
        Int32(cfg.key_dim),
        Int32(cfg.key_heads),
        Int32(width),
        Int32(tokens),
        parts,
    )
    b = view(projected, (qkv_width+z_width+a_width+1):size(projected, 1), :)
    beta = scratch(Float32, size(b))
    beta .= JeffClient.native_sigmoid.(b)
    a = view(projected, (qkv_width+z_width+1):(qkv_width+z_width+a_width), :)
    decay = scratch(Float32, size(a))
    decay .= attention.a_decay .* JeffClient.native_softplus.(a .+ attention.dt_bias)
    z = reshape(
        view(projected, (qkv_width+1):(qkv_width+z_width), :),
        cfg.value_dim,
        cfg.value_heads,
        tokens,
    )
    output = scratch(Float32, (cfg.value_dim, cfg.value_heads, tokens))
    CUDA.@cuda threads=(32, 4) blocks=(cld(cfg.value_dim, 4), cfg.value_heads) delta_recurrent_kernel!(
        output,
        query,
        key,
        mixed,
        beta,
        decay,
        Int32(cfg.key_dim),
        Int32(cfg.value_dim),
        Int32(2width),
        Int32(cfg.value_heads÷cfg.key_heads),
        Int32(tokens),
        parts,
    )
    normalized = JeffClient.native_rms(output, attention.norm, cfg.eps; centered = false)
    gated3 = scratch(Float32, size(normalized))
    gated3 .= normalized .* JeffClient.native_silu.(z)
    gated = reshape(gated3, cfg.value_dim*cfg.value_heads, tokens)
    return JeffClient.native_linear(attention.out, gated)
end
