using JeffClient, BenchmarkTools, LinearAlgebra, Statistics
import AppleAccelerate, JSON

function main()
    length(ARGS) == 2 || error("Usage: sweep_cpu_delta.jl CHECKPOINT REFERENCE_JSON")
    sample = only(JSON.parsefile(ARGS[2]))
    matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    inputs = Dict(name => matrix(rows, Int64) for (name, rows) in sample["inputs"])
    expected = matrix(sample["logits"], Float32)
    backend = NativeBackend(ARGS[1])
    BLAS.set_num_threads(8)
    for chunk in (64, 32, 16, 128), workers in (1, 2, 4, 8)
        withenv(
            "JEFF_CPU_DELTA_CHUNK_SIZE" => string(chunk),
            "JEFF_CPU_DELTA_WORKERS" => string(workers),
            "JEFF_CPU_PARALLEL_HEADS" => "1",
            "JEFF_CPU_DELTA_WORKSPACE" => "1",
            "JEFF_CPU_MLP_WORKSPACE" => "1",
            "JEFF_CPU_VECTOR_MATH" => "1",
            "JEFF_CPU_TRIM_PADDING" => "1",
            "JEFF_CPU_FINAL_QUERY" => "1",
        ) do
            actual = logits(backend, inputs)
            isapprox(actual, expected; atol = 2e-4, rtol = 2e-4) ||
                error("Reference mismatch")
            logits(backend, inputs)
            trial = @benchmark logits($backend, $inputs) samples=5 evals=1 seconds=30
            println(
                JSON.json(
                    Dict(
                        "chunk" => chunk,
                        "workers" => workers,
                        "median_ms" => median(trial).time / 1e6,
                        "p95_ms" => quantile(trial.times, 0.95) / 1e6,
                        "heap_bytes" => median(trial).memory,
                        "allocations" => median(trial).allocs,
                        "max_error" => maximum(abs.(actual .- expected)),
                        "blas_threads" => BLAS.get_num_threads(),
                        "accelerate_threads" => AppleAccelerate.get_num_threads(),
                    ),
                ),
            )
            flush(stdout)
        end
    end
end

main()
