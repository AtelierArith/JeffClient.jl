using JeffClient, LinearAlgebra, BenchmarkTools, Octavian, LoopVectorization, Test

function turbo_block!(output, weight, input, rows)
    @turbo for n in axes(input, 2), m in rows
        value = 0.0f0
        for k in axes(input, 1)
            value += weight[k, m]*input[k, n]
        end
        output[m, n] = value
    end
    return nothing
end

function parallel_product!(output, weight, input, kernel)
    count = min(Threads.nthreads(:default), size(output, 1))
    @sync for worker = 1:count
        rows = (fld((worker-1)*size(output, 1), count)+1):fld(worker*size(output, 1), count)
        Threads.@spawn begin
            if kernel == :blas
                mul!(@view(output[rows, :]), transpose(@view(weight[:, rows])), input)
            elseif kernel == :octavian
                Octavian.matmul_serial!(
                    @view(output[rows, :]),
                    transpose(@view(weight[:, rows])),
                    input,
                )
            else
                turbo_block!(output, weight, input, rows)
            end
        end
    end
    return output
end

function main()
    backend = NativeBackend(only(ARGS))
    BLAS.set_num_threads(1)
    delta = first(l.attention for l in backend.layers if l.attention.kind == :delta)
    for (name, original) in (
            ("gate", first(backend.layers).mlp.gate),
            ("down", first(backend.layers).mlp.down),
            ("qkv", delta.qkv),
        ),
        materialize in (false, true)

        weight = materialize ? Matrix(original) : original
        input = reshape(sin.(Float32.(1:(size(weight, 1)*101))), size(weight, 1), 101)
        output = zeros(Float32, size(weight, 2), 101)
        expected = transpose(weight)*input
        for kernel in (:blas, :octavian, :turbo)
            parallel_product!(output, weight, input, kernel)
            @test output ≈ expected atol=2e-5 rtol=2e-5
            trial =
                @benchmark parallel_product!($output, $weight, $input, $kernel) samples=30 evals=1 seconds=30
            println(
                name,
                " materialize=",
                materialize,
                " kernel=",
                kernel,
                " median_ms=",
                median(trial).time/1e6,
                " bytes=",
                median(trial).memory,
                " max_error=",
                maximum(abs.(output .- expected)),
            )
            flush(stdout)
        end
    end
end
main()
