@inline function warp_sum(value::Float32)
    value += Metal.simd_shuffle_xor(value, Int16(16))
    value += Metal.simd_shuffle_xor(value, Int16(8))
    value += Metal.simd_shuffle_xor(value, Int16(4))
    value += Metal.simd_shuffle_xor(value, Int16(2))
    value += Metal.simd_shuffle_xor(value, Int16(1))
    return value
end

# Each SIMD group owns one value row. A lane keeps its key components in a
# fixed-size register tuple, so reductions require no threadgroup barriers.
function delta_recurrent_kernel!(
    output,
    query,
    key,
    value,
    beta,
    decay,
    key_dim,
    value_dim,
    value_start,
    groups,
    length,
    ::Val{KEY_VALUES},
    ::Val{ROWS},
    ::Val{PRECOMPUTED} = Val(false),
) where {KEY_VALUES,ROWS,PRECOMPUTED}
    local_index = Metal.thread_position_in_threadgroup_2d()
    group_index = Metal.threadgroup_position_in_grid_2d()
    lane = Int32(local_index.x)
    row_in_group = Int32(local_index.y)
    row = (Int32(group_index.x) - Int32(1)) * Int32(ROWS) + row_in_group
    head = Int32(group_index.y)
    key_head = cld(head, groups)
    state = ntuple(_ -> 0.0f0, Val(KEY_VALUES))
    for token = Int32(1):length
        keys = ntuple(Val(KEY_VALUES)) do part
            component = lane + Int32(32 * (part - 1))
            component <= key_dim ? key[component, key_head, token] : 0.0f0
        end
        queries = ntuple(Val(KEY_VALUES)) do part
            component = lane + Int32(32 * (part - 1))
            component <= key_dim ? query[component, key_head, token] : 0.0f0
        end
        factor = PRECOMPUTED ? decay[head, token] : exp(decay[head, token])
        state = map(s -> s * factor, state)
        prediction = warp_sum(sum(map(*, state, keys)))
        v =
            row <= value_dim ? value[value_start+(head-Int32(1))*value_dim+row, token] :
            0.0f0
        correction = (v - prediction) * beta[head, token]
        state = map((s, k) -> s + correction * k, state, keys)
        result = warp_sum(sum(map(*, state, queries)))
        if lane == 1 && row <= value_dim
            output[row, head, token] = result
        end
    end
    return
end

function packed_qk_kernel!(
    output,
    mixed,
    width,
    heads,
    channels,
    columns,
    start,
    factor,
    ::Val{PARTS},
) where {PARTS}
    local_index = Metal.thread_position_in_threadgroup_2d()
    group = Metal.threadgroup_position_in_grid_2d().x
    lane = Int32(local_index.x)
    column = (Int32(group) - Int32(1)) * Int32(8) + Int32(local_index.y)
    if column <= columns
        head = mod(column - Int32(1), heads)
        token = (column - Int32(1)) ÷ heads
        source = token * channels + start + head * width
        destination = (column - Int32(1)) * width
        values = ntuple(Val(PARTS)) do part
            row = lane + Int32(32 * (part - 1))
            row <= width ? (@inbounds mixed[source+row]) : 0.0f0
        end
        squared = 0.0f0
        @inbounds for part = 1:PARTS
            squared += abs2(values[part])
        end
        denominator = sqrt(warp_sum(squared) + 1.0f-6) * factor
        @inbounds for part = 1:PARTS
            row = lane + Int32(32 * (part - 1))
            if row <= width
                output[destination+row] = values[part] / denominator
            end
        end
    end
    return
end

function packed_qk(mixed, cfg, length, start, factor)
    output = pooled_array(Float32, (cfg.key_dim, cfg.key_heads, length))
    columns = cfg.key_heads * length
    Metal.@metal threads=(32, 8) groups=(cld(columns, 8), 1) packed_qk_kernel!(
        output,
        mixed,
        Int32(cfg.key_dim),
        Int32(cfg.key_heads),
        Int32(size(mixed, 1)),
        Int32(columns),
        Int32(start),
        factor,
        Val(cld(cfg.key_dim, 32)),
    )
    return output
end

function packed_qk_pair_kernel!(
    query,
    key,
    mixed,
    width,
    heads,
    channels,
    columns,
    factor,
    parts,
)
    packed_qk_kernel!(
        query,
        mixed,
        width,
        heads,
        channels,
        columns,
        Int32(0),
        factor,
        parts,
    )
    packed_qk_kernel!(
        key,
        mixed,
        width,
        heads,
        channels,
        columns,
        width * heads,
        1.0f0,
        parts,
    )
    return
end

function packed_qk_pair(mixed, cfg, length)
    query = pooled_array(Float32, (cfg.key_dim, cfg.key_heads, length))
    key = pooled_array(Float32, (cfg.key_dim, cfg.key_heads, length))
    columns = cfg.key_heads * length
    Metal.@metal threads=(32, 8) groups=(cld(columns, 8), 1) packed_qk_pair_kernel!(
        query,
        key,
        mixed,
        Int32(cfg.key_dim),
        Int32(cfg.key_heads),
        Int32(size(mixed, 1)),
        Int32(columns),
        sqrt(Float32(cfg.key_dim)),
        Val(cld(cfg.key_dim, 32)),
    )
    return query, key
end

function delta_gates_kernel!(beta, decay, b, a, a_decay, dt_bias)
    heads = Int32(size(a, 1))
    elements = Int32(length(a))
    index = Int(Metal.thread_position_in_grid_1d())
    if index <= elements
        head = (index - 1) % heads + 1
        @inbounds begin
            beta[index] = JeffClient.native_sigmoid(b[index])
            decay[index] =
                a_decay[head] * JeffClient.native_softplus(a[index] + dt_bias[head])
        end
    end
    return
end

function delta_gates(b, a, a_decay, dt_bias)
    size(b) == size(a) || throw(DimensionMismatch("Delta gate projections must match."))
    size(a, 1) == length(a_decay) == length(dt_bias) ||
        throw(DimensionMismatch("Delta gate head counts must match."))
    beta = pooled_array(Float32, size(b))
    decay = pooled_array(Float32, size(a))
    elements = length(a)
    if elements > 0
        launch_cached_kernel!(
            delta_gates_kernel!,
            beta,
            decay,
            b,
            a,
            a_decay,
            dt_bias,
            ;
            threads = 256,
            groups = cld(elements, 256),
        )
    end
    return beta, decay
end

function delta_mask_kernel!(output, input, mask)
    width = Int32(size(input, 1))
    elements = Int32(length(input))
    index = Int32(Metal.thread_position_in_grid_1d())
    if index <= elements
        token = (index - Int32(1)) ÷ width + Int32(1)
        @inbounds output[index] = input[index] * mask[token]
    end
    return
end

function delta_masked_input(input, mask)
    size(input, 2) == length(mask) ||
        throw(DimensionMismatch("Delta mask length must match input columns."))
    output = pooled_array(Float32, size(input))
    elements = length(input)
    if elements > 0
        launch_cached_kernel!(
            delta_mask_kernel!,
            output,
            input,
            mask,
            ;
            threads = 256,
            groups = cld(elements, 256),
        )
    end
    return output
end

function JeffClient.delta_attention(attention, x::Metal.MtlMatrix{Float32}, mask, cfg)
    # Keep the stable chunked implementation available for unsupported widths.
    cfg.key_dim > 256 && return invoke(
        JeffClient.delta_attention,
        Tuple{Any,Any,Any,Any},
        attention,
        x,
        metal_host_mask(mask),
        cfg,
    )
    length = size(x, 2)
    masked = delta_masked_input(x, metal_device_mask(x, mask))
    mixed = JeffClient.causal_depthwise(
        JeffClient.native_linear(attention.qkv, masked),
        attention.conv,
    )
    key_width = cfg.key_dim * cfg.key_heads
    query, key = packed_qk_pair(mixed, cfg, length)
    beta, decay = delta_gates(
        JeffClient.native_linear(attention.b, masked),
        JeffClient.native_linear(attention.a, masked),
        attention.a_decay,
        attention.dt_bias,
    )
    z = reshape(
        JeffClient.native_linear(attention.z, masked),
        cfg.value_dim,
        cfg.value_heads,
        length,
    )
    output = pooled_array(Float32, (cfg.value_dim, cfg.value_heads, length))
    key_values = cld(cfg.key_dim, 32)
    rows = 8
    Metal.@metal threads=(32, rows) groups=(cld(cfg.value_dim, rows), cfg.value_heads) delta_recurrent_kernel!(
        output,
        query,
        key,
        mixed,
        beta,
        decay,
        Int32(cfg.key_dim),
        Int32(cfg.value_dim),
        Int32(2key_width),
        Int32(cfg.value_heads ÷ cfg.key_heads),
        Int32(length),
        Val(key_values),
        Val(rows),
    )
    gated = rms_silu_gate(output, z, attention.norm, cfg.eps)
    return JeffClient.native_linear(
        attention.out,
        reshape(gated, cfg.value_dim * cfg.value_heads, length),
    )
end
