# Like Laya's split_rope, prepare the batched-MPS layout directly. Jeff also
# needs per-head RMS, partial RoPE, grouped KV heads, and the separate Q gate.
struct RopeTables
    key::Tuple{Int,Int,Float32}
    cosines::Metal.MtlArray{Float32,2,Metal.SharedStorage}
    sines::Metal.MtlArray{Float32,2,Metal.SharedStorage}
end

# Keep only the current table pair per queue. Replaced tables remain rooted by
# queued kernels; shared storage is not rewritten until GPU work completes.
const ROPE_TABLE_CACHE = Dict{UInt,RopeTables}()
const ROPE_TABLE_LOCK = ReentrantLock()

function rope_tables(reference, cfg, length)
    queue = objectid(Metal.global_queue(Metal.device()))
    key = (cfg.rotary_dim, length, cfg.rope_theta)
    @lock ROPE_TABLE_LOCK begin
        cached = get(ROPE_TABLE_CACHE, queue, nothing)
        if cached === nothing || cached.key != key
            half = cfg.rotary_dim ÷ 2
            # Preserve the CPU implementation's Float32 frequency arithmetic.
            frequency = inv.(
                cfg.rope_theta .^
                (Float32.(0:2:(cfg.rotary_dim-1)) ./ Float32(cfg.rotary_dim)),
            )
            theta = frequency .* permutedims(Float32.(0:(length-1)))
            cached = RopeTables(
                key,
                JeffClient.on_native_device(reference, reshape(cos.(theta), half, length)),
                JeffClient.on_native_device(reference, reshape(sin.(theta), half, length)),
            )
            ROPE_TABLE_CACHE[queue] = cached
        end
        return cached.cosines, cached.sines
    end
end

function normalized_rope_kernel!(
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
    half,
    eps,
    ::Val{PARTS},
    ::Val{QUERY},
) where {PARTS,QUERY}
    local_index = Metal.thread_position_in_threadgroup_2d()
    group = Metal.threadgroup_position_in_grid_2d().x
    lane = Int32(local_index.x)
    column = (Int32(group) - Int32(1)) * Int32(8) + Int32(local_index.y)
    if column <= length * heads
        token = rem(column - Int32(1), length) + Int32(1)
        head = div(column - Int32(1), length) + Int32(1)
        input_head = div(head - Int32(1), div(heads, input_heads)) + Int32(1)
        stride = QUERY ? Int32(2) * width : width
        input_offset = stride * ((token - Int32(1)) * input_heads + input_head - Int32(1))
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

function prepare_query(projection, weight, tables, cfg, length)
    size(projection) == (2cfg.head_dim * cfg.heads, length) &&
    Base.length(weight) == cfg.head_dim || throw(
        DimensionMismatch(
            "Query projection and RMS weights must match the configured heads.",
        ),
    )
    output = pooled_array(Float32, (cfg.head_dim, length, cfg.heads))
    Metal.@metal threads=(32, 8) groups=(cld(length * cfg.heads, 8), 1) normalized_rope_kernel!(
        output,
        output,
        projection,
        projection,
        weight,
        tables[1],
        tables[2],
        Int32(cfg.head_dim),
        Int32(cfg.heads),
        Int32(cfg.heads),
        Int32(length),
        Int32(cfg.rotary_dim ÷ 2),
        cfg.eps,
        Val(cld(cfg.head_dim, 32)),
        Val(true),
    )
    return output
end

function prepare_key_value(key_projection, value_projection, weight, tables, cfg, length)
    size(key_projection) ==
    size(value_projection) ==
    (cfg.head_dim * cfg.kv_heads, length) && Base.length(weight) == cfg.head_dim || throw(
        DimensionMismatch(
            "KV projections and RMS weights must match the configured heads.",
        ),
    )
    key = pooled_array(Float32, (cfg.head_dim, length, cfg.heads))
    value = pooled_array(Float32, size(key))
    Metal.@metal threads=(32, 8) groups=(cld(length * cfg.heads, 8), 1) normalized_rope_kernel!(
        key,
        value,
        key_projection,
        value_projection,
        weight,
        tables[1],
        tables[2],
        Int32(cfg.head_dim),
        Int32(cfg.heads),
        Int32(cfg.kv_heads),
        Int32(length),
        Int32(cfg.rotary_dim ÷ 2),
        cfg.eps,
        Val(cld(cfg.head_dim, 32)),
        Val(false),
    )
    return key, value
end

function merge_gate_kernel!(output, values, qgate, width, heads, length)
    index = Metal.thread_position_in_grid_2d()
    row, token = Int32(index.x), Int32(index.y)
    if row <= width * heads && token <= length
        head = div(row - Int32(1), width) + Int32(1)
        channel = rem(row - Int32(1), width) + Int32(1)
        source = channel + width * (token - Int32(1) + length * (head - Int32(1)))
        gate_source =
            channel +
            width +
            Int32(2) * width * (head - Int32(1) + heads * (token - Int32(1)))
        @inbounds output[row, token] =
            values[source] * JeffClient.native_sigmoid(qgate[gate_source])
    end
    return
end

function merge_gate(values, qgate, cfg, length)
    size(values) == (cfg.head_dim, length, cfg.heads) &&
    size(qgate) == (2cfg.head_dim * cfg.heads, length) || throw(
        DimensionMismatch("Attention values and gates must match the configured heads."),
    )
    output = pooled_array(Float32, (cfg.head_dim * cfg.heads, length))
    launch_cached_kernel!(
        merge_gate_kernel!,
        output,
        values,
        qgate,
        Int32(cfg.head_dim),
        Int32(cfg.heads),
        Int32(length);
        threads = (64, 4),
        groups = (cld(size(output, 1), 64), cld(length, 4)),
    )
    return output
end
