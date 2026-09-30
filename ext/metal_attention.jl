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
    cfg.head_dim > 4096 && return invoke(
        JeffClient.full_attention,
        Tuple{Any,Any,Any,Any},
        attention,
        x,
        metal_host_mask(mask),
        cfg,
    )
    0 <= cfg.rotary_dim <= cfg.head_dim && iseven(cfg.rotary_dim) ||
        throw(ArgumentError("Rotary width must be even and within the head width."))
    length = size(x, 2)
    tables = rope_tables(x, cfg, length)
    qgate = JeffClient.native_linear(attention.q, x)
    query = prepare_query(qgate, attention.q_norm, tables, cfg, length)
    key, value = prepare_key_value(
        JeffClient.native_linear(attention.k, x),
        JeffClient.native_linear(attention.v, x),
        attention.k_norm,
        tables,
        cfg,
        length,
    )
    scores = head_matmul(key, query, 'T', 'N')
    probabilities = masked_softmax(scores, mask, cfg.head_dim)
    values = head_matmul(value, probabilities, 'N', 'N')
    return JeffClient.native_linear(attention.out, merge_gate(values, qgate, cfg, length))
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
    launch_cached_kernel!(
        causal_depthwise_kernel!,
        output,
        input,
        weight,
        Int32(channels),
        Int32(length),
        Int32(size(weight, 1));
        threads = (64, 4),
        groups = (cld(channels, 64), cld(length, 4)),
    )
    return output
end

# Columns are contiguous sequences: sample b occupies ((b-1)*length+1):(b*length).
# A tap may read earlier columns within its sample, never a previous sample.
function batched_causal_depthwise_kernel!(
    output,
    input,
    weight,
    channels,
    columns,
    sequence_length,
    kernel,
)
    index = Metal.thread_position_in_grid_2d()
    channel, token = Int32(index.x), Int32(index.y)
    if channel <= channels && token <= columns
        first_token = ((token - Int32(1)) ÷ sequence_length) * sequence_length + Int32(1)
        value = 0.0f0
        for tap = Int32(1):kernel
            source_token = token - (kernel - tap)
            if source_token >= first_token
                value += input[channel, source_token] * weight[tap, channel]
            end
        end
        output[channel, token] = JeffClient.native_silu(value)
    end
    return
end

function batched_causal_depthwise(input::Metal.MtlMatrix{Float32}, weight, sequence_length)
    channels, columns = size(input)
    sequence_length > 0 && columns > 0 && columns % sequence_length == 0 ||
        throw(ArgumentError("Batch columns must contain complete nonempty sequences."))
    size(weight, 2) == channels && size(weight, 1) > 0 ||
        throw(DimensionMismatch("Convolution weights must match input channels."))
    output = pooled_array(Float32, size(input))
    launch_cached_kernel!(
        batched_causal_depthwise_kernel!,
        output,
        input,
        weight,
        Int32(channels),
        Int32(columns),
        Int32(sequence_length),
        Int32(size(weight, 1));
        threads = (64, 4),
        groups = (cld(channels, 64), cld(columns, 4)),
    )
    return output
end
