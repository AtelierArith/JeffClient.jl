# Batched MPSGraph products use (row, column, head) arrays. Column-major
# tensor conversion reverses these dimensions for MPSGraph automatically.
function head_matmul(
    a::Metal.MtlArray{Float32,3},
    b::Metal.MtlArray{Float32,3},
    transpose_a,
    transpose_b,
)
    rows = size(a, transpose_a == 'N' ? 1 : 2)
    inner_a = size(a, transpose_a == 'N' ? 2 : 1)
    inner_b = size(b, transpose_b == 'N' ? 1 : 2)
    columns = size(b, transpose_b == 'N' ? 2 : 1)
    inner_a == inner_b && size(a, 3) == size(b, 3) ||
        throw(DimensionMismatch("Head matrix dimensions must match."))
    result = pooled_array(Float32, (rows, columns, size(a, 3)))
    return batched_matmul!(result, a, b, transpose_a, transpose_b)
end

function JeffClient.full_attention(attention, x::Metal.MtlMatrix{Float32}, mask, cfg)
    length = size(x, 2)
    qgate =
        reshape(JeffClient.native_linear(attention.q, x), 2cfg.head_dim, cfg.heads, length)
    query = JeffClient.native_rope(
        JeffClient.native_rms(qgate[1:cfg.head_dim, :, :], attention.q_norm, cfg.eps),
        cfg,
    )
    key = JeffClient.native_rope(
        JeffClient.native_rms(
            reshape(
                JeffClient.native_linear(attention.k, x),
                cfg.head_dim,
                cfg.kv_heads,
                length,
            ),
            attention.k_norm,
            cfg.eps,
        ),
        cfg,
    )
    value = reshape(
        JeffClient.native_linear(attention.v, x),
        cfg.head_dim,
        cfg.kv_heads,
        length,
    )
    groups = cfg.heads ÷ cfg.kv_heads
    kv_heads =
        JeffClient.on_native_device(x, Int32[cld(head, groups) for head = 1:cfg.heads])
    query = permutedims(query, (1, 3, 2))
    key = permutedims(key[:, kv_heads, :], (1, 3, 2))
    value = permutedims(value[:, kv_heads, :], (1, 3, 2))
    scores = head_matmul(key, query, 'T', 'N')
    probabilities = masked_softmax(scores, mask, cfg.head_dim)
    values = permutedims(head_matmul(value, probabilities, 'N', 'N'), (1, 3, 2))
    gated = values .* JeffClient.native_sigmoid.(qgate[(cfg.head_dim+1):end, :, :])
    return JeffClient.native_linear(
        attention.out,
        reshape(gated, cfg.head_dim * cfg.heads, length),
    )
end

function causal_depthwise_kernel!(output, input, weight, channels, length, kernel)
    index = Metal.thread_position_in_grid_2d()
    channel, token = Int32(index.x), Int32(index.y)
    if channel <= channels && token <= length
        value = 0.0f0
        for tap = Int32(1):kernel
            source_token = token - (kernel - tap)
            if source_token >= 1
                value += input[channel, source_token] * weight[tap, channel]
            end
        end
        output[channel, token] = JeffClient.native_silu(value)
    end
    return
end

function JeffClient.causal_depthwise(input::Metal.MtlMatrix{Float32}, weight)
    channels, length = size(input)
    output = pooled_array(Float32, size(input))
    Metal.@metal threads=(64, 4) groups=(cld(channels, 64), cld(length, 4)) causal_depthwise_kernel!(
        output,
        input,
        weight,
        Int32(channels),
        Int32(length),
        Int32(size(weight, 1)),
    )
    return output
end
