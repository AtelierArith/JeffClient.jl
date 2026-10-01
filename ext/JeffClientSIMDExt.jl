module JeffClientSIMDExt

import JeffClient
using SIMD
using LinearAlgebra: Transpose

function JeffClient.cpu_convolution!(
    output::Matrix{Float32},
    input::Matrix{Float32},
    weight::Transpose{Float32,Matrix{Float32}},
)
    JeffClient.cpu_setting(:simd) || return invoke(
        JeffClient.cpu_convolution!,
        Tuple{Any,Any,Any},
        output,
        input,
        weight,
    )
    channels, tokens = size(input)
    coefficients = parent(weight)
    size(output) == size(input) ||
        throw(DimensionMismatch("Convolution output shape differs."))
    size(coefficients, 1) == channels ||
        throw(DimensionMismatch("Convolution channels differ."))
    kernel = size(coefficients, 2)
    width = 8
    stop = channels - mod(channels, width)
    GC.@preserve output input coefficients begin
        for token = 1:tokens
            for channel = 1:width:stop
                value = Vec{8,Float32}(0)
                for tap = 1:kernel
                    source = token - (kernel - tap)
                    source < 1 && continue
                    a = vload(
                        Vec{8,Float32},
                        pointer(input, (source - 1) * channels + channel),
                    )
                    b = vload(
                        Vec{8,Float32},
                        pointer(coefficients, (tap - 1) * channels + channel),
                    )
                    value = value + a * b
                end
                vstore(value, pointer(output, (token - 1) * channels + channel))
            end
            @inbounds for channel = (stop+1):channels
                value = 0.0f0
                for tap = 1:kernel
                    source = token - (kernel - tap)
                    source < 1 && continue
                    value += input[channel, source] * coefficients[channel, tap]
                end
                output[channel, token] = value
            end
        end
    end
    return output
end

end
