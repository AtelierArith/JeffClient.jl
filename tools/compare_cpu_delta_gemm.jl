using LinearAlgebra, BenchmarkTools, Octavian, Test
import AppleAccelerate

function main()
    BLAS.set_num_threads(8)
    q = reshape(sin.(Float32.(1:(128*16*101))), 128, 16, 101)
    k = cos.(q)
    state = reshape(sin.(Float32.(1:(128*128))), 128, 128)
    for n in (37, 64)
        qc, kc = @view(q[:, 3, 1:n]), @view(k[:, 3, 1:n])
        for (name, a, b) in (
            ("system", transpose(kc), qc),
            ("state_query", state, qc),
            ("state_update", qc, transpose(kc)),
        )
            expected = a * b
            output = similar(expected)
            for (label, operation) in (
                ("accelerate", () -> mul!(output, a, b)),
                ("octavian_serial", () -> Octavian.matmul_serial!(output, a, b)),
            )
                operation()
                @test output ≈ expected atol=2e-5 rtol=2e-5
                trial = @benchmark $operation() samples=100 evals=1 seconds=5
                println(
                    name,
                    " n=",
                    n,
                    " ",
                    label,
                    " median_us=",
                    median(trial).time/1e3,
                    " bytes=",
                    median(trial).memory,
                    " max_error=",
                    maximum(abs.(output .- expected)),
                )
                flush(stdout)
            end
        end
    end
end

main()
