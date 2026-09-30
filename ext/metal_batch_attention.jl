# Sample-local RoPE positions, with heads*batch as the MPS batch axis.
function batched_normalized_rope_kernel!(
    output,
    value_output,
    projection,
    value_projection,
    weight,
    cosines,
    sines,
    width,
    heads,
    input_heads,
    length,
    batch,
    half,
    eps,
    ::Val{PARTS},
    ::Val{QUERY},
) where {PARTS,QUERY}
    local_index = Metal.thread_position_in_threadgroup_2d()
    group = Metal.threadgroup_position_in_grid_2d().x
    lane = Int32(local_index.x)
    column = (Int32(group) - Int32(1)) * Int32(8) + Int32(local_index.y)
    if column <= length * heads * batch
        token = rem(column - Int32(1), length) + Int32(1)
        head_sample = div(column - Int32(1), length)
        head = rem(head_sample, heads) + Int32(1)
        input_token = token + div(head_sample, heads) * length
        input_head = div(head - Int32(1), div(heads, input_heads)) + Int32(1)
        stride = QUERY ? Int32(2) * width : width
        input_offset =
            stride * ((input_token - Int32(1)) * input_heads + input_head - Int32(1))
        output_offset = (column - Int32(1)) * width
        values = ntuple(Val(PARTS)) do part
            row = lane + Int32(32 * (part - 1))
            row <= width ? (@inbounds projection[input_offset+row]) : 0.0f0
        end
        squared = 0.0f0
        @inbounds for part = 1:PARTS
            squared += abs2(values[part])
        end
        inverse = inv(sqrt(warp_sum(squared) / Float32(width) + eps))
        @inbounds for part = 1:PARTS
            row = lane + Int32(32 * (part - 1))
            if row <= width
                normalized = values[part] * inverse * (1.0f0 + weight[row])
                if row <= Int32(2) * half
                    first = row <= half
                    pair = first ? row + half : row - half
                    partner =
                        projection[input_offset+pair] * inverse * (1.0f0 + weight[pair])
                    table_index = (token - Int32(1)) * half + (first ? row : pair)
                    c, s = cosines[table_index], sines[table_index]
                    normalized =
                        first ? normalized * c - partner * s : normalized * c + partner * s
                end
                output[output_offset+row] = normalized
                if !QUERY
                    value_output[output_offset+row] = value_projection[input_offset+row]
                end
            end
        end
    end
    return
end

function batched_prepare_heads(
    projection,
    value_projection,
    weight,
    tables,
    cfg,
    sequence_length,
    ::Val{QUERY},
) where {QUERY}
    columns = size(projection, 2)
    sequence_length > 0 && columns > 0 && columns % sequence_length == 0 ||
        throw(ArgumentError("Batch projections must contain complete sequences."))
    input_heads = QUERY ? cfg.heads : cfg.kv_heads
    cfg.heads > 0 && input_heads > 0 && cfg.heads % input_heads == 0 ||
        throw(ArgumentError("Grouped attention heads must divide query heads."))
    0 < cfg.head_dim <= 4096 &&
    0 <= cfg.rotary_dim <= cfg.head_dim &&
    iseven(cfg.rotary_dim) ||
        throw(ArgumentError("Unsupported batched attention head or rotary width."))
    stride = QUERY ? 2cfg.head_dim : cfg.head_dim
    size(projection, 1) == stride * input_heads && length(weight) == cfg.head_dim || throw(
        DimensionMismatch("Batch projections and normalization weights must match heads."),
    )
    QUERY ||
        size(value_projection) == size(projection) ||
        throw(DimensionMismatch("Batch key and value projections must match."))
    batch = columns ÷ sequence_length
    output = pooled_array(Float32, (cfg.head_dim, sequence_length, cfg.heads * batch))
    values = QUERY ? output : pooled_array(Float32, size(output))
    if cfg.head_dim == 256
        launch_cached_kernel!(
            batched_normalized_rope_kernel!,
            output,
            values,
            projection,
            value_projection,
            weight,
            tables[1],
            tables[2],
            Int32(cfg.head_dim),
            Int32(cfg.heads),
            Int32(input_heads),
            Int32(sequence_length),
            Int32(batch),
            Int32(cfg.rotary_dim ÷ 2),
            cfg.eps,
            Val(8),
            Val(QUERY);
            threads = (32, 8),
            groups = (cld(sequence_length * cfg.heads * batch, 8), 1),
        )
    else
        threads = (32, 8)
        groups = (cld(sequence_length * cfg.heads * batch, 8), 1)
        Metal.@metal threads=threads groups=groups batched_normalized_rope_kernel!(
            output,
            values,
            projection,
            value_projection,
            weight,
            tables[1],
            tables[2],
            Int32(cfg.head_dim),
            Int32(cfg.heads),
            Int32(input_heads),
            Int32(sequence_length),
            Int32(batch),
            Int32(cfg.rotary_dim ÷ 2),
            cfg.eps,
            Val(cld(cfg.head_dim, 32)),
            Val(QUERY),
        )
    end
    return output, values
end

function batched_merge_gate_kernel!(
    output,
    values,
    qgate,
    width,
    heads,
    sequence_length,
    columns,
)
    index = Metal.thread_position_in_grid_2d()
    row, token = Int32(index.x), Int32(index.y)
    if row <= width * heads && token <= columns
        head = div(row - Int32(1), width)
        channel = rem(row - Int32(1), width) + Int32(1)
        sample = div(token - Int32(1), sequence_length)
        local_token = rem(token - Int32(1), sequence_length)
        source = channel + width * (local_token + sequence_length * (head + heads * sample))
        gate_source =
            channel + width + Int32(2) * width * (head + heads * (token - Int32(1)))
        @inbounds output[row, token] =
            values[source] * JeffClient.native_sigmoid(qgate[gate_source])
    end
    return
end

function batched_merge_gate(values, qgate, cfg, sequence_length)
    columns = size(qgate, 2)
    sequence_length > 0 && columns > 0 && columns % sequence_length == 0 ||
        throw(ArgumentError("Batch gates must contain complete sequences."))
    size(values) ==
    (cfg.head_dim, sequence_length, cfg.heads * (columns ÷ sequence_length)) &&
    size(qgate, 1) == 2cfg.head_dim * cfg.heads ||
        throw(DimensionMismatch("Batched attention values and gates must match."))
    output = pooled_array(Float32, (cfg.head_dim * cfg.heads, columns))
    launch_cached_kernel!(
        batched_merge_gate_kernel!,
        output,
        values,
        qgate,
        Int32(cfg.head_dim),
        Int32(cfg.heads),
        Int32(sequence_length),
        Int32(columns);
        threads = (64, 4),
        groups = (cld(size(output, 1), 64), cld(columns, 4)),
    )
    return output
end

function batched_full_attention(attention, x, mask, cfg, sequence_length)
    tables = rope_tables(x, cfg, sequence_length)
    qgate = JeffClient.native_linear(attention.q, x)
    query, _ = batched_prepare_heads(
        qgate,
        qgate,
        attention.q_norm,
        tables,
        cfg,
        sequence_length,
        Val(true),
    )
    key, value = batched_prepare_heads(
        JeffClient.native_linear(attention.k, x),
        JeffClient.native_linear(attention.v, x),
        attention.k_norm,
        tables,
        cfg,
        sequence_length,
        Val(false),
    )
    scores = head_matmul(key, query, 'T', 'N')
    probabilities = batched_masked_softmax(scores, mask, cfg.head_dim, cfg.heads)
    values = head_matmul(value, probabilities, 'N', 'N')
    return JeffClient.native_linear(
        attention.out,
        batched_merge_gate(values, qgate, cfg, sequence_length),
    )
end
