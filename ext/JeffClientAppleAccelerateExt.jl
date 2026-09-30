module JeffClientAppleAccelerateExt

import JeffClient
import AppleAccelerate

function JeffClient.native_cpu_silu!(output::Matrix{Float32}, exponential::Matrix{Float32})
    if !Sys.isapple() || get(ENV, "JEFF_CPU_VECTOR_MATH", "0") != "1"
        return invoke(JeffClient.native_cpu_silu!, Tuple{Any,Any}, output, exponential)
    end
    size(output) == size(exponential) ||
        throw(DimensionMismatch("SiLU scratch shapes differ."))
    exponential .= .-output
    AppleAccelerate.exp!(exponential, exponential)
    output .= output ./ (1.0f0 .+ exponential)
    return output
end

function JeffClient.cpu_owned_mlp_gate!(gate::Matrix{Float32}, up::Matrix{Float32})
    if !Sys.isapple() || get(ENV, "JEFF_CPU_VECTOR_MATH", "0") != "1"
        return invoke(JeffClient.cpu_owned_mlp_gate!, Tuple{Any,Any}, gate, up)
    end
    size(gate) == size(up) || throw(DimensionMismatch("MLP gate shapes differ."))
    # Check before consuming either array: multiplying first can overflow even
    # when SiLU(gate) * up is finite. Preserve the scalar expression in that case.
    @inbounds for index in eachindex(gate, up)
        product = gate[index] * up[index]
        if isfinite(gate[index]) && isfinite(up[index]) && !isfinite(product)
            return invoke(JeffClient.cpu_owned_mlp_gate!, Tuple{Any,Any}, gate, up)
        end
    end
    up .*= gate
    gate .= .-gate
    AppleAccelerate.exp!(gate, gate)
    gate .= up ./ (1.0f0 .+ gate)
    return gate
end

end
