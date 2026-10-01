module JeffClientLoopVectorizationExt

import JeffClient
using LoopVectorization

# Guard the fast-math loop against exceptional values, signed zeros, and tiny
# outputs. Outside this conservative domain use the unchanged scalar formula.
function eligible_silu(values)
    valid = true
    @inbounds @simd for i in eachindex(values)
        x = values[i]
        valid &= isfinite(x) & (x > -20.0f0) & (x < 80.0f0) & (abs(x) > 1.0f-12)
    end
    return valid
end

function eligible_up(values)
    valid = true
    @inbounds @simd for i in eachindex(values)
        x = abs(values[i])
        valid &= isfinite(x) & (x > 1.0f-12) & (x < 1.0f12)
    end
    return valid
end

function JeffClient.cpu_portable_silu!(output::Matrix{Float32})
    get(ENV, "JEFF_CPU_PORTABLE_VECTOR_MATH", "0") == "1" || return false
    isempty(output) && return true
    eligible_silu(output) || return false
    @turbo for i in eachindex(output)
        output[i] = output[i] * inv(1.0f0 + exp(-output[i]))
    end
    return true
end

function JeffClient.cpu_portable_gate!(gate::Matrix{Float32}, up::Matrix{Float32})
    get(ENV, "JEFF_CPU_PORTABLE_VECTOR_MATH", "0") == "1" || return false
    size(gate) == size(up) || return false
    Base.mightalias(gate, up) && return false
    isempty(gate) && return true
    eligible_silu(gate) && eligible_up(up) || return false
    @turbo for i in eachindex(gate, up)
        gate[i] = (gate[i] * inv(1.0f0 + exp(-gate[i]))) * up[i]
    end
    return true
end

end
