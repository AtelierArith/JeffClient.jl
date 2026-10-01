module JeffClientLoopVectorizationExt

import JeffClient
using LoopVectorization

# Guard the fast-math loop against exceptional values, signed zeros, and tiny
# outputs. Outside this conservative domain use the unchanged scalar formula.
function eligible_silu(values, span = eachindex(values))
    valid = true
    @inbounds @simd for i in span
        x = values[i]
        valid &= isfinite(x) & (x > -20.0f0) & (x < 80.0f0) & (abs(x) > 1.0f-12)
    end
    return valid
end

function JeffClient.cpu_portable_softmax!(scores::Matrix{Float32})
    JeffClient.cpu_setting(:portable_vector_math) || return false
    isempty(scores) && return false
    valid = true
    @inbounds @simd for i in eachindex(scores)
        valid &= isfinite(scores[i])
    end
    valid || return false
    # Preserve rare subnormal probabilities with the scalar exp/division;
    # SIMD lanes use bounded inputs, including exact zero for masked scores.
    indices = Vector{Int}(undef, size(scores, 1))
    differences = Vector{Float32}(undef, size(scores, 1))
    for column in axes(scores, 2)
        largest = -Inf32
        @turbo for row in axes(scores, 1)
            largest = max(largest, scores[row, column])
        end
        count = 0
        @inbounds for row in axes(scores, 1)
            difference = scores[row, column] - largest
            if -104.0f0 <= difference < -80.0f0
                count += 1
                indices[count] = row
                differences[count] = difference
            end
        end
        @turbo for row in axes(scores, 1)
            difference = scores[row, column] - largest
            scores[row, column] =
                ifelse(difference >= -80.0f0, exp(max(difference, -80.0f0)), 0.0f0)
        end
        total = 0.0f0
        @turbo for row in axes(scores, 1)
            total += scores[row, column]
        end
        @turbo for row in axes(scores, 1)
            scores[row, column] /= total
        end
        @inbounds for j = 1:count
            scores[indices[j], column] = exp(differences[j]) / total
        end
    end
    return true
end

function eligible_up(values, span = eachindex(values))
    valid = true
    @inbounds @simd for i in span
        x = abs(values[i])
        valid &= isfinite(x) & (x > 1.0f-12) & (x < 1.0f12)
    end
    return valid
end

function zero_block(values, span)
    valid = true
    @inbounds @simd for i in span
        valid &= iszero(values[i])
    end
    return valid
end

# Leave exceptional lanes untouched in the SIMD pass. Their original values
# are then available for the scalar formula, including NaN and signed zero.
function mixed_silu_block!(output, first, last, exceptions)
    count = 0
    @inbounds for i = first:last
        x = output[i]
        if !(x > -20.0f0 && x < 80.0f0 && abs(x) > 1.0f-12)
            count += 1
            exceptions[count] = i
        end
    end
    @turbo for i = first:last
        x = output[i]
        valid = (x > -20.0f0) & (x < 80.0f0) & (abs(x) > 1.0f-12)
        safe = ifelse(valid, x, 1.0f0)
        output[i] = ifelse(valid, safe * inv(1.0f0 + exp(-safe)), x)
    end
    @inbounds for j = 1:count
        i = exceptions[j]
        x = output[i]
        output[i] = iszero(x) ? x : JeffClient.native_silu(x)
    end
    return output
end

function mixed_gate_block!(gate, up, first, last, exceptions)
    count = 0
    @inbounds for i = first:last
        x, u = gate[i], abs(up[i])
        if !(x > -20.0f0 && x < 80.0f0 && abs(x) > 1.0f-12 && u > 1.0f-12 && u < 1.0f12)
            count += 1
            exceptions[count] = i
        end
    end
    @turbo for i = first:last
        x, u = gate[i], up[i]
        valid =
            (x > -20.0f0) & (x < 80.0f0) & (abs(x) > 1.0f-12) & (abs(u) > 1.0f-12) &
            (abs(u) < 1.0f12)
        safe_x, safe_u = ifelse(valid, x, 1.0f0), ifelse(valid, u, 1.0f0)
        gate[i] = ifelse(valid, (safe_x * inv(1.0f0 + exp(-safe_x))) * safe_u, x)
    end
    @inbounds for j = 1:count
        i = exceptions[j]
        x = gate[i]
        gate[i] = (iszero(x) ? x : JeffClient.native_silu(x)) * up[i]
    end
    return gate
end

function silu_blocks!(output, starts)
    exceptions = Vector{Int}(undef, 256)
    for first in starts
        last = min(first + 255, length(output))
        if eligible_silu(output, first:last)
            @turbo for i = first:last
                output[i] = output[i] * inv(1.0f0 + exp(-output[i]))
            end
        elseif !zero_block(output, first:last)
            mixed_silu_block!(output, first, last, exceptions)
        end
    end
    return output
end

function gate_blocks!(gate, up, starts, exceptions = Vector{Int}(undef, 256))
    for first in starts
        last = min(first + 255, length(gate))
        if eligible_silu(gate, first:last) && eligible_up(up, first:last)
            @turbo for i = first:last
                gate[i] = (gate[i] * inv(1.0f0 + exp(-gate[i]))) * up[i]
            end
        elseif zero_block(gate, first:last)
            @inbounds @simd for i = first:last
                gate[i] = (gate[i] * 0.5f0) * up[i]
            end
        else
            mixed_gate_block!(gate, up, first, last, exceptions)
        end
    end
    return gate
end

function JeffClient.cpu_portable_silu!(output::Matrix{Float32})
    JeffClient.cpu_setting(:portable_vector_math) || return false
    isempty(output) && return true
    blocks = JeffClient.cpu_setting(:vector_math_blocks)
    workers = min(Threads.nthreads(:default), cld(length(output), 65536))
    if blocks && workers > 1
        # Interleave disjoint blocks so a long zero prefix cannot leave most
        # workers idle. Workers use owned data and never change BLAS settings.
        @sync for worker = 1:workers
            Threads.@spawn silu_blocks!(
                output,
                (1+256*(worker-1)):(256*workers):length(output),
            )
        end
        return true
    end
    if !eligible_silu(output)
        blocks || return false
        silu_blocks!(output, 1:256:length(output))
        return true
    end
    @turbo for i in eachindex(output)
        output[i] = output[i] * inv(1.0f0 + exp(-output[i]))
    end
    return true
end

function JeffClient.cpu_portable_gate!(gate::Matrix{Float32}, up::Matrix{Float32})
    JeffClient.cpu_setting(:portable_vector_math) || return false
    size(gate) == size(up) || return false
    Base.mightalias(gate, up) && return false
    isempty(gate) && return true
    blocks = JeffClient.cpu_setting(:vector_math_blocks)
    workers = min(Threads.nthreads(:default), cld(length(gate), 65536))
    if blocks && workers > 1
        @sync for worker = 1:workers
            Threads.@spawn gate_blocks!(
                gate,
                up,
                (1+256*(worker-1)):(256*workers):length(gate),
            )
        end
        return true
    end
    if !(eligible_silu(gate) && eligible_up(up))
        blocks || return false
        gate_blocks!(gate, up, 1:256:length(gate))
        return true
    end
    @turbo for i in eachindex(gate, up)
        gate[i] = (gate[i] * inv(1.0f0 + exp(-gate[i]))) * up[i]
    end
    return true
end

end
