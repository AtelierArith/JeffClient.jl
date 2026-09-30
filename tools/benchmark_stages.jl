using BenchmarkTools
using JeffClient
using LinearAlgebra
import JSON
import Metal

function completed_call(f, args)
    result = f(args...)
    Metal.synchronize()
    return result
end

function measure_stage(name, f, args)
    completed_call(f, args)
    completed_call(f, args)
    GC.gc(true)
    trial = @benchmark completed_call($f, $args) samples=10 evals=1 seconds=30
    measured = BenchmarkTools.median(trial)
    return Dict(
        "stage" => name,
        "median_ms" => measured.time / 1e6,
        "julia_heap_bytes" => measured.memory,
        "julia_allocations" => measured.allocs,
    )
end

function mlp_projection(mlp, x)
    gate = JeffClient.native_silu.(JeffClient.native_linear(mlp.gate, x))
    return JeffClient.native_linear(mlp.down, gate .* JeffClient.native_linear(mlp.up, x))
end

function main()
    length(ARGS) in (2, 3) || error(
        "Usage: julia --project=tools tools/benchmark_stages.jl CHECKPOINT REFERENCE_JSON [OUTPUT_JSON]",
    )
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    BLAS.set_num_threads(8)
    references = JSON.parsefile(ARGS[2])
    cases = references isa AbstractDict ? references["cases"] : references
    sample = first(cases)
    ids = Int64.(first(sample["inputs"]["input_ids"]))
    mask = Int64.(first(sample["inputs"]["attention_mask"]))
    backend = NativeBackend(ARGS[1]; device = :metal)
    hidden = JeffClient.native_gather(backend.embedding, ids)
    delta = first(layer for layer in backend.layers if layer.attention.kind == :delta)
    full = first(layer for layer in backend.layers if layer.attention.kind == :full)
    mixed = JeffClient.native_linear(delta.attention.qkv, hidden)
    # Isolated, synchronized stage costs diagnose kernels/submission overhead.
    # They are not additive: complete forwards batch operations differently.
    results = [
        measure_stage(name, f, args) for (name, f, args) in (
            (
                "DeltaNet attention",
                JeffClient.delta_attention,
                (delta.attention, hidden, mask, backend.config),
            ),
            (
                "full attention",
                JeffClient.full_attention,
                (full.attention, hidden, mask, backend.config),
            ),
            ("MLP projections", mlp_projection, (delta.mlp, hidden)),
            (
                "RMS normalization",
                JeffClient.native_rms,
                (hidden, delta.input_norm, backend.config.eps),
            ),
            (
                "causal depthwise convolution",
                JeffClient.causal_depthwise,
                (mixed, delta.attention.conv),
            ),
            ("QKV projection", JeffClient.native_linear, (delta.attention.qkv, hidden)),
        )
    ]
    report = Dict(
        "device" => string(Metal.device().name),
        "sequence_length" => length(ids),
        "samples_per_stage" => 10,
        "stages" => results,
    )
    println(JSON.json(report, 2))
    if length(ARGS) == 3
        mkpath(dirname(abspath(ARGS[3])))
        write(ARGS[3], JSON.json(report, 2) * "\n")
    end
end

main()
