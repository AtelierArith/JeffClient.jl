using JeffClient, LinearAlgebra, BenchmarkTools, Octavian, Test
import AppleAccelerate
if get(ENV, "JEFF_CPU_BLIS", "0") == "1"
    import BLISBLAS
end

function main()
    backend = NativeBackend(only(ARGS))
    BLAS.set_num_threads(8)
    if get(ENV, "JEFF_CPU_BLIS", "0") == "1"
        BLISBLAS.set_num_threads(8)
        println("BLIS threads: ", BLISBLAS.get_num_threads())
    end
    println("BLAS: ", BLAS.get_config(), "; threads=", BLAS.get_num_threads())
    for (name, weight) in
        (("gate", first(backend.layers).mlp.gate), ("down", first(backend.layers).mlp.down))
        input = reshape(sin.(Float32.(1:(size(weight, 1)*101))), size(weight, 1), 101)
        output = zeros(Float32, size(weight, 2), 101)
        expected = transpose(weight) * input
        packed = permutedims(weight)
        right = zeros(Float32, size(input, 2), size(weight, 2))
        for (label, operation) in (
            ("blas", () -> mul!(output, transpose(weight), input)),
            ("octavian", () -> Octavian.matmul!(output, transpose(weight), input)),
            (
                "octavian_serial",
                () -> Octavian.matmul_serial!(output, transpose(weight), input),
            ),
            ("packed_weight", () -> mul!(output, packed, input)),
            ("right_and_copy", () -> begin
                mul!(right, transpose(input), weight)
                permutedims!(output, right, (2, 1))
            end),
            (
                "octavian_right_and_copy",
                () -> begin
                    Octavian.matmul!(right, transpose(input), weight)
                    permutedims!(output, right, (2, 1))
                end,
            ),
        )
            operation()
            @test output ≈ expected atol=2e-5 rtol=2e-5
            trial = @benchmark $operation() samples=20 evals=1 seconds=20
            println(
                name,
                " ",
                label,
                " median_ms=",
                median(trial).time/1e6,
                " bytes=",
                median(trial).memory,
                " error=",
                maximum(abs.(output .- expected)),
            )
            flush(stdout)
        end
    end
end

main()
