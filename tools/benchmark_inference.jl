using BenchmarkTools
using JeffClient
using LinearAlgebra
using Profile
import JSON
if get(ENV, "JEFF_CPU_OCTAVIAN_DELTA", "0") == "1"
    import Octavian
    Base.get_extension(JeffClient, :JeffClientOctavianExt) === nothing &&
        error("Octavian extension was not loaded; run Pkg.resolve() before benchmarking.")
end
if get(ENV, "JEFF_CPU_SIMD", "0") == "1"
    import SIMD
    Base.get_extension(JeffClient, :JeffClientSIMDExt) === nothing &&
        error("SIMD extension was not loaded; run Pkg.resolve() before benchmarking.")
end

if get(ENV, "JEFF_CPU_ACCELERATE", "0") == "1"
    Sys.isapple() || error("Apple Accelerate requires macOS.")
    import AppleAccelerate
    any(lib -> occursin("Accelerate", lib.libname), BLAS.get_config().loaded_libs) ||
        error("Accelerate BLAS forwarding requires macOS 13.4 or later.")
end


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
    BLAS.set_num_threads(parse(Int, get(ENV, "JEFF_BLAS_THREADS", "8")))
    # Apply backend overrides after general BLAS setup. Accelerate 0.7 only
    # selects single-threaded (1) or automatic multithreading (anything else).
    if get(ENV, "JEFF_CPU_ACCELERATE", "0") == "1" &&
       haskey(ENV, "JEFF_CPU_ACCELERATE_THREADS")
        AppleAccelerate.set_num_threads(parse(Int, ENV["JEFF_CPU_ACCELERATE_THREADS"]))
    end
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
            "blas_config" => string(BLAS.get_config()),
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
        if device == :cpu
            result["cpu_recurrent_delta_enabled"] =
                get(ENV, "JEFF_CPU_RECURRENT_DELTA", "0") == "1"
            result["cpu_octavian_delta_enabled"] =
                get(ENV, "JEFF_CPU_OCTAVIAN_DELTA", "0") == "1"
            result["cpu_final_query_enabled"] = get(ENV, "JEFF_CPU_FINAL_QUERY", "0") == "1"
            result["cpu_simd_enabled"] = get(ENV, "JEFF_CPU_SIMD", "0") == "1"
            result["cpu_parallel_heads_enabled"] =
                get(ENV, "JEFF_CPU_PARALLEL_HEADS", "0") == "1"
            result["julia_worker_threads"] = Threads.nthreads(:default)
            result["cpu_mlp_workspace_enabled"] =
                get(ENV, "JEFF_CPU_MLP_WORKSPACE", "0") == "1"
            result["cpu_delta_workspace_enabled"] =
                get(ENV, "JEFF_CPU_DELTA_WORKSPACE", "0") == "1"
            result["cpu_delta_chunk_size"] = JeffClient.cpu_delta_chunk_size()
            result["cpu_delta_workers"] = JeffClient.cpu_delta_workers(backend.config)
            result["cpu_inplace_delta_rms_enabled"] =
                get(ENV, "JEFF_CPU_INPLACE_DELTA_RMS", "0") == "1"
            result["cpu_vector_math_enabled"] = get(ENV, "JEFF_CPU_VECTOR_MATH", "0") == "1"
            result["cpu_accelerate_requested"] = get(ENV, "JEFF_CPU_ACCELERATE", "0") == "1"
            if result["cpu_accelerate_requested"]
                result["accelerate_version"] = string(pkgversion(AppleAccelerate))
                result["accelerate_threads"] = AppleAccelerate.get_num_threads()
            end
            result["cpu_trim_padding_enabled"] =
                get(ENV, "JEFF_CPU_TRIM_PADDING", "0") == "1"
            result["cpu_computed_sequence_lengths"] = [
                size(inputs["input_ids"], 2) - JeffClient.native_sequence_start(
                    backend.embedding,
                    inputs["attention_mask"],
                    row,
                ) + 1 for row in axes(inputs["input_ids"], 1)
            ]
        end
        if device == :metal
            result["metal_batched_enabled"] = get(ENV, "JEFF_METAL_BATCHED", "0") == "1"
            result["metal_workspace_enabled"] = get(ENV, "JEFF_METAL_WORKSPACE", "0") == "1"
            result["metal_packed_mlp_enabled"] =
                get(ENV, "JEFF_METAL_PACKED_MLP", "0") == "1"
            result["metal_trim_padding_enabled"] =
                get(ENV, "JEFF_METAL_TRIM_PADDING", "0") == "1"
            result["metal_fused_delta_mask_enabled"] =
                get(ENV, "JEFF_METAL_FUSED_DELTA_MASK", "0") == "1"
            result["metal_shape_workspaces_enabled"] =
                get(ENV, "JEFF_METAL_SHAPE_WORKSPACES", "0") == "1"
            result["metal_computed_sequence_lengths"] = [
                size(inputs["input_ids"], 2) - JeffClient.native_sequence_start(
                    backend.embedding,
                    inputs["attention_mask"],
                    row,
                ) + 1 for row in axes(inputs["input_ids"], 1)
            ]
            batched_execution =
                result["metal_batched_enabled"] &&
                size(inputs["input_ids"], 1) > 1 &&
                backend.config.key_dim <= 256 &&
                backend.config.head_dim <= 4096 &&
                size(backend.embedding, 1) <= 4096
            result["metal_batched_execution"] = batched_execution
            if batched_execution
                fill!(
                    result["metal_computed_sequence_lengths"],
                    maximum(result["metal_computed_sequence_lengths"]),
                )
            end
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
