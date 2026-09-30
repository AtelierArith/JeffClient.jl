@inline function warp_max(value::Float32)
    value = max(value, Metal.simd_shuffle_xor(value, Int16(16)))
    value = max(value, Metal.simd_shuffle_xor(value, Int16(8)))
    value = max(value, Metal.simd_shuffle_xor(value, Int16(4)))
    value = max(value, Metal.simd_shuffle_xor(value, Int16(2)))
    value = max(value, Metal.simd_shuffle_xor(value, Int16(1)))
    return value
end

# Following Laya's per-SIMD-column reduction, fuse scale, causal/padding mask,
# maximum, exp, sum, and normalization. No dense host mask or scratch arrays.
function masked_softmax_kernel!(output, scores, mask, length, columns, scale)
    local_index = Metal.thread_position_in_threadgroup_2d()
    group = Metal.threadgroup_position_in_grid_2d().x
    lane = Int32(local_index.x)
    column = (Int32(group) - Int32(1)) * Int32(8) + Int32(local_index.y)
    if column <= columns
        query = mod(column - Int32(1), length) + Int32(1)
        offset = (column - Int32(1)) * length
        largest = -Inf32
        @inbounds for key = lane:Int32(32):length
            value =
                key <= query && mask[key] == 1.0f0 ? scores[offset+key] * scale :
                -floatmax(Float32)
            largest = max(largest, value)
        end
        largest = warp_max(largest)
        total = 0.0f0
        @inbounds for key = lane:Int32(32):length
            value =
                key <= query && mask[key] == 1.0f0 ? scores[offset+key] * scale :
                -floatmax(Float32)
            total += exp(value - largest)
        end
        total = warp_sum(total)
        @inbounds for key = lane:Int32(32):length
            value =
                key <= query && mask[key] == 1.0f0 ? scores[offset+key] * scale :
                -floatmax(Float32)
            output[offset+key] = exp(value - largest) / total
        end
    end
    return
end

function masked_softmax(scores::Metal.MtlArray{Float32,3}, mask, head_dim)
    length = size(scores, 1)
    device_mask = metal_device_mask(scores, mask)
    length == size(scores, 2) == Base.length(device_mask) ||
        throw(DimensionMismatch("Attention mask dimensions must match."))
    columns = Base.length(scores) ÷ length
    output = pooled_array(Float32, size(scores))
    Metal.@metal threads=(32, 8) groups=(cld(columns, 8), 1) masked_softmax_kernel!(
        output,
        scores,
        device_mask,
        Int32(length),
        Int32(columns),
        inv(sqrt(Float32(head_dim))),
    )
    return output
end
