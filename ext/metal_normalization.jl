# One logical 32-lane SIMD group normalizes a column. Fixed-size tuples keep
# inputs in registers through the reduction and output, without scratch arrays.
function normalization_kernel!(
    output,
    input,
    weight,
    eps,
    factor,
    width,
    columns,
    ::Val{PARTS},
    ::Val{MEAN},
    ::Val{CENTERED},
    ::Val{WEIGHTED},
) where {PARTS,MEAN,CENTERED,WEIGHTED}
    local_index = Metal.thread_position_in_threadgroup_2d()
    group = Metal.threadgroup_position_in_grid_2d().x
    lane = Int32(local_index.x)
    column = (Int32(group) - Int32(1)) * Int32(8) + Int32(local_index.y)
    if column <= columns
        offset = (column - Int32(1)) * width
        values = ntuple(Val(PARTS)) do part
            row = lane + Int32(32 * (part - 1))
            row <= width ? (@inbounds input[offset+row]) : 0.0f0
        end
        # Laya uses an explicit accumulator here. Base's tuple map/sum at
        # 32 elements crashed LLVM's Metal inliner on the real hidden width.
        squared = 0.0f0
        @inbounds for part = 1:PARTS
            squared += abs2(values[part])
        end
        squared = warp_sum(squared)
        variance = MEAN ? squared / Float32(width) : squared
        denominator = sqrt(variance + eps) * factor
        inverse = inv(denominator)
        @inbounds for part = 1:PARTS
            row = lane + Int32(32 * (part - 1))
            if row <= width
                normalized = MEAN ? values[part] * inverse : values[part] / denominator
                if WEIGHTED
                    w = CENTERED ? 1.0f0 + weight[row] : weight[row]
                    normalized *= w
                end
                output[offset+row] = normalized
            end
        end
    end
    return
end

function fused_normalization(x, weight, eps, factor, mean, centered, weighted)
    width = size(x, 1)
    columns = length(x) ÷ width
    output = pooled_array(Float32, size(x))
    Metal.@metal threads=(32, 8) groups=(cld(columns, 8), 1) normalization_kernel!(
        output,
        x,
        weight,
        Float32(eps),
        Float32(factor),
        Int32(width),
        Int32(columns),
        Val(cld(width, 32)),
        mean,
        centered,
        weighted,
    )
    return output
end

function JeffClient.native_rms(x::Metal.MtlArray{Float32}, weight, eps; centered = true)
    length(weight) == size(x, 1) ||
        throw(DimensionMismatch("RMS weight width must match the input."))
    size(x, 1) > 4096 &&
        return invoke(JeffClient.native_rms, Tuple{Any,Any,Any}, x, weight, eps; centered)
    # Split the Bool explicitly: Val(centered) alone leaves a runtime Val type
    # at this keyword boundary and caused JET dispatch reports in the forward.
    if centered
        return fused_normalization(x, weight, eps, 1.0f0, Val(true), Val(true), Val(true))
    else
        return fused_normalization(x, weight, eps, 1.0f0, Val(true), Val(false), Val(true))
    end
end

function l2_normalize(x::Metal.MtlArray{Float32}, factor)
    size(x, 1) > 4096 && return x ./ (sqrt.(sum(abs2, x; dims = 1) .+ 1.0f-6) .* factor)
    # The unweighted specialization never reads this dummy weight argument.
    return fused_normalization(x, x, 1.0f-6, factor, Val(false), Val(false), Val(false))
end
