using JeffClient, LinearAlgebra, BenchmarkTools, Test

function separate!(outputs, weights, input)
    for (output, weight) in zip(outputs, weights)
        JeffClient.cpu_projection!(output, weight, input)
    end
    return outputs
end

function fused_copy!(outputs, combined, weight, input)
    JeffClient.cpu_projection!(combined, weight, input)
    offset = 0
    for output in outputs
        copyto!(output, @view combined[(offset+1):(offset+size(output, 1)), :])
        offset += size(output, 1)
    end
    return outputs
end

function main()
    backend = NativeBackend(only(ARGS))
    BLAS.set_num_threads(1)
    first_delta =
        first(layer.attention for layer in backend.layers if layer.attention.kind == :delta)
    groups = (
        gate_up = (first(backend.layers).mlp.gate, first(backend.layers).mlp.up),
        qkv_z = (first_delta.qkv, first_delta.z),
    )
    withenv(
        "JEFF_CPU_PARALLEL_PROJECTIONS" => "1",
        "JEFF_CPU_PROJECTION_THREAD_SCOPE" => "1",
    ) do
        JeffClient.cpu_projection_scope() do
            for (name, weights) in pairs(groups), n in (1, 101, 256)
                input = reshape(
                    sin.(Float32.(1:(size(first(weights), 1)*n))),
                    size(first(weights), 1),
                    n,
                )
                outputs = map(w -> zeros(Float32, size(w, 2), n), weights)
                expected = map(w -> transpose(w)*input, weights)
                combined_weight = hcat(weights...)
                combined = zeros(Float32, size(combined_weight, 2), n)
                for mode in (:separate, :direct, :copy)
                    operation =
                        mode == :copy ?
                        () -> fused_copy!(outputs, combined, combined_weight, input) :
                        (
                            mode == :direct ?
                            () -> JeffClient.cpu_projection!(
                                combined,
                                combined_weight,
                                input,
                            ) : () -> separate!(outputs, weights, input)
                        )
                    operation()
                    if mode == :direct
                        @test combined ≈ vcat(expected...) atol=2e-5 rtol=2e-5
                    else
                        @test all(
                            isapprox(o, e; atol = 2e-5, rtol = 2e-5) for
                            (o, e) in zip(outputs, expected)
                        )
                    end
                    trial = @benchmark $operation() samples=30 evals=1 seconds=20
                    println(
                        name,
                        " tokens=",
                        n,
                        " mode=",
                        mode,
                        " median_ms=",
                        median(trial).time/1e6,
                        " bytes=",
                        median(trial).memory,
                    )
                    flush(stdout)
                end
            end
        end
    end
end
main()
