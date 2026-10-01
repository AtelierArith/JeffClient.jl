using JeffClient, LinearAlgebra, BenchmarkTools, Test
if JeffClient.cpu_setting(:accelerate)
    import AppleAccelerate
end

function block_product!(output, weight, input, rows)
    lhs =
        weight isa Transpose{Float32,Matrix{Float32}} ? @view(parent(weight)[rows, :]) :
        transpose(@view(weight[:, rows]))
    mul!(@view(output[rows, :]), lhs, input)
    return nothing
end

function parallel_product!(output, weight, input, workers)
    count = min(workers, size(output, 1))
    @sync for worker = 1:count
        rows = (fld((worker-1)*size(output, 1), count)+1):fld(worker*size(output, 1), count)
        Threads.@spawn block_product!(output, weight, input, rows)
    end
    return output
end

function main()
    backend = NativeBackend(only(ARGS))
    BLAS.set_num_threads(1)
    JeffClient.cpu_setting(:accelerate) && AppleAccelerate.set_num_threads(1)
    println("BLAS ", BLAS.get_config(), "; threads=", BLAS.get_num_threads())
    for (name, original) in (
            ("gate", first(backend.layers).mlp.gate),
            ("down", first(backend.layers).mlp.down),
            ("qkv", first(backend.layers).attention.qkv),
        ),
        packed in (false, true)

        weight = packed ? transpose(permutedims(original)) : original
        input = reshape(sin.(Float32.(1:(size(weight, 1)*101))), size(weight, 1), 101)
        output = zeros(Float32, size(weight, 2), 101)
        expected = transpose(weight) * input
        for workers in (1, 2, 4, 8, 12, 16)
            workers > Threads.nthreads(:default) && continue
            operation =
                workers == 1 ? () -> mul!(output, transpose(weight), input) :
                () -> parallel_product!(output, weight, input, workers)
            operation()
            @test output ≈ expected atol=2e-5 rtol=2e-5
            trial = @benchmark $operation() samples=20 evals=1 seconds=20
            println(
                name,
                " packed=",
                packed,
                " workers=",
                workers,
                " median_ms=",
                median(trial).time/1e6,
                " bytes=",
                median(trial).memory,
            )
            flush(stdout)
        end
    end
end

main()
