module JeffClientOctavianExt

import JeffClient, Octavian
using LinearAlgebra: mul!

function JeffClient.cpu_delta_state_product!(
    output::Matrix{Float32},
    state::Matrix{Float32},
    rhs::AbstractMatrix{Float32},
    alpha::Float32,
    beta::Float32,
)
    if JeffClient.cpu_setting(:octavian_delta) &&
       size(state, 1) <= 256 &&
       size(state, 2) <= 256 &&
       size(rhs, 2) <= 128
        # No nested Julia or BLAS thread pool inside an independently owned worker.
        return Octavian.matmul_serial!(output, state, rhs, alpha, beta)
    end
    return mul!(output, state, rhs, alpha, beta)
end

end
