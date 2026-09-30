using BenchmarkTools
using JeffClient
using LinearAlgebra
using Profile
import JSON

if length(ARGS) >= 2 && ARGS[2] == "metal"
    import Metal
end

function synchronized_logits(backend::NativeBackend, inputs, device)
    result = logits(backend, inputs)
    device == :metal && Metal.synchronize()
    return result
end

function synchronized_logits(backend::ONNXBackend, inputs, device)
    return backend.session(inputs, [backend.output_name])[backend.output_name]
end

function main()
    length(ARGS) in 3:6 || error(
        "Usage: julia --project=tools tools/benchmark_inference.jl MODEL cpu|metal|onnx REFERENCE_JSON [CASE_INDEX] [SAMPLES] [OUTPUT_JSON]",
    )
    device = Symbol(ARGS[2])
    device in (:cpu, :metal, :onnx) || error("Choose cpu, metal, or onnx.")
    index = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    samples = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 10
    samples >= 3 || error("Use at least three measurement samples.")
    output = length(ARGS) >= 6 ? ARGS[6] : nothing
    if device == :metal
        Metal.functional() || error("A functional Apple GPU is required.")
        Metal.allowscalar(false)
    end
    BLAS.set_num_threads(8)
    references = JSON.parsefile(ARGS[3])
    cases = references isa AbstractDict ? references["cases"] : references
    sample = cases[index]
    rows_to_matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    inputs = Dict(name => rows_to_matrix(rows, Int64) for (name, rows) in sample["inputs"])
    expected = rows_to_matrix(sample["logits"], Float32)
    model_load = @elapsed backend =
        device == :onnx ? load_export(ARGS[1]) : NativeBackend(ARGS[1]; device)
    try
        first_call = @elapsed actual = synchronized_logits(backend, inputs, device)
        max_error = maximum(abs.(actual .- expected))
        all(abs.(actual .- expected) .<= 2.0f-4 .+ 2.0f-4 .* abs.(expected)) ||
            error("Scores do not match the independent reference.")
        # Warm separately from the cold measurement, then measure exactly one
        # complete forward per evaluation. Model loading and input conversion
        # are excluded; final score transfer and GPU synchronization are included.
        synchronized_logits(backend, inputs, device)
        GC.gc(true)
        trial =
            @benchmark synchronized_logits($backend, $inputs, $device) samples=samples evals=1 seconds=120
        median_estimate = BenchmarkTools.median(trial)
        sorted_ms = sort(trial.times ./ 1e6)
        result = Dict(
            "backend" => String(device),
            "julia_version" => string(VERSION),
            "machine" => Sys.MACHINE,
            "cpu" => Sys.cpu_info()[1].model,
            "device" => device == :metal ? string(Metal.device().name) : "CPU",
            "blas_threads" => BLAS.get_num_threads(),
            "batch_size" => size(inputs["input_ids"], 1),
            "sequence_length" => size(inputs["input_ids"], 2),
            "active_tokens" => vec(sum(inputs["attention_mask"]; dims = 2)),
            "samples" => length(trial.times),
            "model_load_seconds" => model_load,
            "first_forward_seconds" => first_call,
            "median_ms" => median_estimate.time / 1e6,
            "minimum_ms" => minimum(sorted_ms),
            "maximum_ms" => maximum(sorted_ms),
            "p95_ms" => sorted_ms[ceil(Int, 0.95 * length(sorted_ms))],
            "median_julia_heap_bytes" => median_estimate.memory,
            "median_julia_allocations" => median_estimate.allocs,
            "max_logit_error" => max_error,
        )
        if device == :metal
            result["metal_workspace_enabled"] = get(ENV, "JEFF_METAL_WORKSPACE", "0") == "1"
            result["metal_packed_mlp_enabled"] =
                get(ENV, "JEFF_METAL_PACKED_MLP", "0") == "1"
            extension = Base.get_extension(JeffClient, :JeffClientMetalExt)
            isdefined(extension, :metal_pool_stats) &&
                (result["metal_pool"] = extension.metal_pool_stats())
            # A post-trial snapshot depends on when finalizers have run.
            # Keep it, then report a second snapshot after explicit collection
            # and synchronization; neither operation is part of trial timing.
            GC.gc(true)
            Metal.synchronize()
            isdefined(extension, :recycle_uploads!) && extension.recycle_uploads!()
            isdefined(extension, :metal_pool_stats) &&
                (result["metal_pool_after_gc"] = extension.metal_pool_stats())
            if isdefined(extension, :trim_completed_buffer_pool!)
                extension.trim_completed_buffer_pool!()
                result["metal_pool_after_trim"] = extension.metal_pool_stats()
            end
        end
        println(JSON.json(result, 2))
        if output !== nothing
            mkpath(dirname(abspath(output)))
            write(output, JSON.json(result, 2) * "\n")
        end
        if get(ENV, "JEFF_PROFILE", "0") == "1"
            Profile.clear()
            Profile.@profile synchronized_logits(backend, inputs, device)
            Profile.print(; format = :flat, sortedby = :count, mincount = 10)
        end
    finally
        device == :onnx && close(backend)
    end
end

main()
