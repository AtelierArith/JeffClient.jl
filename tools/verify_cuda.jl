using CUDA, JeffClient, Test
using QwenDecisionCore
const JSON = QwenDecisionCore.JSON

function main()
    length(ARGS) in (2, 3) ||
        error("Usage: verify_cuda.jl CHECKPOINT REFERENCE_JSON [GPU=1]")
    gpu = length(ARGS) == 3 ? parse(Int, ARGS[3]) : min(1, length(CUDA.devices())-1)
    CUDA.device!(gpu)
    CUDA.allowscalar(false)
    backend = NativeBackend(ARGS[1]; device = :cuda)
    source = JSON.parsefile(ARGS[2])
    cases = source isa AbstractDict ? source["cases"] : source
    matrix(rows, T) = permutedims(hcat([T.(row) for row in rows]...))
    @testset "Real CUDA checkpoint shape/mask/GC verification" begin
        for trim in ("0", "1")
            withenv("JEFF_CUDA_TRIM_PADDING"=>trim) do
                for case in cases
                    inputs =
                        Dict(name=>matrix(rows, Int64) for (name, rows) in case["inputs"])
                    saved = deepcopy(inputs)
                    expected = matrix(case["logits"], Float32)
                    actual = JeffClient.logits(backend, inputs)
                    @test size(actual) == size(expected)
                    @test actual ≈ expected atol=2e-4 rtol=2e-4
                    retained = copy(actual)
                    GC.gc(true)
                    @test JeffClient.logits(backend, inputs) ≈ expected atol=2e-4 rtol=2e-4
                    @test actual == retained
                    @test inputs == saved
                    println(
                        "trim=",
                        trim,
                        " case=",
                        get(case, "name", "unnamed"),
                        " max error=",
                        maximum(abs.(actual .- expected)),
                    )
                    flush(stdout)
                end
            end
        end
    end
end
main()
