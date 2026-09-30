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
) where {KEY_VALUES,ROWS}
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
        factor = exp(decay[head, token])
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
    masked = x .* reshape(metal_device_mask(x, mask), 1, :)
    mixed = JeffClient.causal_depthwise(
        JeffClient.native_linear(attention.qkv, masked),
        attention.conv,
    )
    key_width = cfg.key_dim * cfg.key_heads
    query = packed_qk(mixed, cfg, length, 0, sqrt(Float32(cfg.key_dim)))
    key = packed_qk(mixed, cfg, length, key_width, 1.0f0)
    beta = JeffClient.native_sigmoid.(JeffClient.native_linear(attention.b, masked))
    decay =
        attention.a_decay .* JeffClient.native_softplus.(
            JeffClient.native_linear(attention.a, masked) .+ attention.dt_bias,
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
