using JeffClient, SIMD, LoopVectorization, BenchmarkTools, Test
using QwenDecisionCore

# Trial only: do not replace inference kernels based on a microbenchmark.
function turbo_convolution!(output, input, coefficients)
    channels, tokens = size(input)
    kernel = size(coefficients, 2)
    fill!(output, 0)
    for tap = 1:kernel
        lag = kernel - tap
        lag >= tokens && continue
        @turbo for token = (lag+1):tokens, channel = 1:channels
            output[channel, token] += input[channel, token-lag] * coefficients[channel, tap]
        end
    end
    return output
end

function main()
    input = reshape(sin.(Float32.(1:(6144*101))), 6144, 101)
    coefficients = reshape(cos.(Float32.(1:(6144*4))), 6144, 4)
    weight = transpose(coefficients)
    output = zeros(Float32, size(input))
    expected = copy(output)
    QwenDecisionCore.cpu_convolution!(expected, input, weight)
    for method in (:scalar_simd, :explicit_simd, :turbo)
        QwenDecisionCore.with_cpu_settings(
            :simd => (method == :explicit_simd ? "1" : "0"),
        ) do
            operation =
                method == :turbo ? () -> turbo_convolution!(output, input, coefficients) :
                () -> (
                    fill!(output, 0);
                    QwenDecisionCore.cpu_convolution!(output, input, weight)
                )
            operation()
            error = maximum(abs.(output .- expected))
            @test output ≈ expected atol=2.0f-6 rtol=2.0f-6
            trial = @benchmark $operation() samples=100 evals=1 seconds=2
            println(
                method,
                ": median_ns=",
                median(trial).time,
                "; bytes=",
                median(trial).memory,
                "; max_error=",
                error,
            )
        end
    end
end

main()
