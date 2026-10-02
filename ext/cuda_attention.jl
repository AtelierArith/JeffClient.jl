function prepare_head_kernel!(
    output,
    projected,
    norm,
    dim,
    heads,
    source_heads,
    tokens,
    rotary,
    theta,
    eps,
    query,
)
    lane = Int32(CUDA.threadIdx().x)
    group = Int32(CUDA.blockIdx().x)-Int32(1)
    head = rem(group, heads)+Int32(1)
    token = group÷heads+Int32(1)
    source_head = query ? head : cld(head, heads÷source_heads)
    source_width = query ? Int32(2)*dim : dim
    base = (source_head-Int32(1))*source_width
    total = 0.0f0
    @inbounds for row = lane:Int32(32):dim
        value = projected[base+row, token]
        total += value*value
    end
    scale = inv(sqrt(warp_sum(total)/Float32(dim)+eps))
    half = rotary÷Int32(2)
    @inbounds for row = lane:Int32(32):dim
        value = projected[base+row, token]*scale*(1.0f0+norm[row])
        if row <= rotary
            first_row = row <= half ? row : row-half
            partner = row <= half ? row+half : row-half
            other = projected[base+partner, token]*scale*(1.0f0+norm[partner])
            angle =
                Float32(token-Int32(1)) /
                theta^(Float32(Int32(2)*(first_row-Int32(1)))/Float32(rotary))
            sine, cosine = sincos(angle)
            value = row <= half ? value*cosine-other*sine : value*cosine+other*sine
        end
        output[row, token, head] = value
    end
    return
end

function prepare_value_kernel!(output, input, dim, heads, source_heads, tokens)
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(
            CUDA.threadIdx().x,
        )
    if index <= dim*tokens*heads
        row = rem(index-Int32(1), dim)+Int32(1)
        column = (index-Int32(1))÷dim
        token = rem(column, tokens)+Int32(1)
        head = column÷tokens+Int32(1)
        source_head = cld(head, heads÷source_heads)
        @inbounds output[index] = input[(source_head-Int32(1))*dim+row, token]
    end
    return
end

function attention_softmax_kernel!(scores, mask, tokens, scale)
    lane = Int32(CUDA.threadIdx().x)
    query = Int32(CUDA.blockIdx().x)
    head = Int32(CUDA.blockIdx().y)
    peak = -floatmax(Float32)
    @inbounds for key = lane:Int32(32):tokens
        value =
            key <= query && mask[key] == 1 ? scores[key, query, head]*scale :
            -floatmax(Float32)
        scores[key, query, head] = value
        peak = max(peak, value)
    end
    for offset in (16, 8, 4, 2, 1)
        peak = max(peak, CUDA.shfl_xor_sync(CUDA.FULL_MASK, peak, offset))
    end
    total = 0.0f0
    @inbounds for key = lane:Int32(32):tokens
        value = exp(scores[key, query, head]-peak)
        scores[key, query, head] = value
        total += value
    end
    total = warp_sum(total)
    @inbounds for key = lane:Int32(32):tokens
        scores[key, query, head] /= total
    end
    return
end

function merge_gate_kernel!(output, values, qgate, dim, heads, tokens)
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(
            CUDA.threadIdx().x,
        )
    if index <= dim*heads*tokens
        row = rem(index-Int32(1), dim)+Int32(1)
        column = (index-Int32(1))÷dim
        head = rem(column, heads)+Int32(1)
        token = column÷heads+Int32(1)
        @inbounds output[index] =
            values[row, token, head]*JeffClient.native_sigmoid(
                qgate[(head-Int32(1))*Int32(2)*dim+dim+row, token],
            )
    end
    return
end

function JeffClient.full_attention(attention, x::CUDA.CuMatrix{Float32}, mask, cfg)
    tokens = size(x, 2)
    dim, heads, kv_heads = cfg.head_dim, cfg.heads, cfg.kv_heads
    qgate = JeffClient.native_linear(attention.q, x)
    kp = JeffClient.native_linear(attention.k, x)
    vp = JeffClient.native_linear(attention.v, x)
    q = scratch(Float32, (dim, tokens, heads))
    k = scratch(Float32, size(q))
    v = scratch(Float32, size(q))
    CUDA.@cuda threads=32 blocks=heads*tokens prepare_head_kernel!(
        q,
        qgate,
        attention.q_norm,
        Int32(dim),
        Int32(heads),
        Int32(heads),
        Int32(tokens),
        Int32(cfg.rotary_dim),
        cfg.rope_theta,
        cfg.eps,
        true,
    )
    CUDA.@cuda threads=32 blocks=heads*tokens prepare_head_kernel!(
        k,
        kp,
        attention.k_norm,
        Int32(dim),
        Int32(heads),
        Int32(kv_heads),
        Int32(tokens),
        Int32(cfg.rotary_dim),
        cfg.rope_theta,
        cfg.eps,
        false,
    )
    CUDA.@cuda threads=256 blocks=cld(length(v), 256) prepare_value_kernel!(
        v,
        vp,
        Int32(dim),
        Int32(heads),
        Int32(kv_heads),
        Int32(tokens),
    )
    scores = scratch(Float32, (tokens, tokens, heads))
    batched_matmul!(scores, k, q, 'T', 'N')
    device_mask = mask isa CUDA.CuArray ? mask : JeffClient.native_prepare_mask(x, mask)
    CUDA.@cuda threads=32 blocks=(tokens, heads) attention_softmax_kernel!(
        scores,
        device_mask,
        Int32(tokens),
        inv(sqrt(Float32(dim))),
    )
    values = scratch(Float32, size(q))
    batched_matmul!(values, v, scores, 'N', 'N')
    merged = scratch(Float32, (dim*heads, tokens))
    CUDA.@cuda threads=256 blocks=cld(length(merged), 256) merge_gate_kernel!(
        merged,
        values,
        qgate,
        Int32(dim),
        Int32(heads),
        Int32(tokens),
    )
    return JeffClient.native_linear(attention.out, merged)
end
