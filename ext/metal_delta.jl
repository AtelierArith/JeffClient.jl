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
        v = row <= value_dim ? value[row, head, token] : 0.0f0
        correction = (v - prediction) * beta[head, token]
        state = map((s, k) -> s + correction * k, state, keys)
        result = warp_sum(sum(map(*, state, queries)))
        if lane == 1 && row <= value_dim
            output[row, head, token] = result
        end
    end
    return
end

function JeffClient.delta_attention(attention, x::Metal.MtlMatrix{Float32}, mask, cfg)
    # Keep the stable chunked implementation available for unsupported widths.
    cfg.key_dim > 256 && return invoke(
        JeffClient.delta_attention,
        Tuple{Any,Any,Any,Any},
        attention,
        x,
        mask,
        cfg,
    )
    length = size(x, 2)
    masked = x .* JeffClient.on_native_device(x, reshape(Float32.(mask), 1, :))
    mixed = JeffClient.causal_depthwise(
        JeffClient.native_linear(attention.qkv, masked),
        attention.conv,
    )
    key_width = cfg.key_dim * cfg.key_heads
    query = reshape(mixed[1:key_width, :], cfg.key_dim, cfg.key_heads, length)
    key = reshape(mixed[(key_width+1):2key_width, :], cfg.key_dim, cfg.key_heads, length)
    value = reshape(mixed[(2key_width+1):end, :], cfg.value_dim, cfg.value_heads, length)
    query = l2_normalize(query, sqrt(Float32(cfg.key_dim)))
    key = l2_normalize(key, 1.0f0)
    beta = JeffClient.native_sigmoid.(JeffClient.native_linear(attention.b, masked))
    decay =
        -exp.(attention.a_log) .* JeffClient.native_softplus.(
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
        value,
        beta,
        decay,
        Int32(cfg.key_dim),
        Int32(cfg.value_dim),
        Int32(cfg.value_heads ÷ cfg.key_heads),
        Int32(length),
        Val(key_values),
        Val(rows),
    )
    normalized = JeffClient.native_rms(output, attention.norm, cfg.eps; centered = false)
    gated = normalized .* JeffClient.native_silu.(z)
    return JeffClient.native_linear(
        attention.out,
        reshape(gated, cfg.value_dim * cfg.value_heads, length),
    )
end
