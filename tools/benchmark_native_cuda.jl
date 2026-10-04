# CUDA and JeffClient must be installed in the active application environment.
using CUDA, JeffClient, Statistics
using QwenDecisionCore
const JSON = QwenDecisionCore.JSON

function main()
    3 <= length(ARGS) <= 5 || error(
        "Usage: benchmark_native_cuda.jl CHECKPOINT REFERENCE_JSON OUTPUT_JSON [SAMPLES=30] [GPU=0]",
    )
    checkpoint, reference, output = ARGS[1:3]
    samples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
    device = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
    samples >= 3 || error("Use at least three samples.")
    CUDA.device!(device)
    CUDA.allowscalar(false)
    load_seconds = @elapsed backend = NativeBackend(checkpoint; device = :cuda)
    source = JSON.parsefile(reference)
    cases = source isa AbstractDict ? source["cases"] : source
    matrix(rows, T) = permutedims(hcat([T.(row) for row in rows]...))
    inputs_for(case) = Dict(name=>matrix(rows, Int64) for (name, rows) in case["inputs"])
    errors = Float64[]
    for case in cases
        expected = matrix(case["logits"], Float32)
        actual = JeffClient.logits(backend, inputs_for(case))
        size(actual) == size(expected) || error("Output shape mismatch")
        all(isapprox.(actual, expected; atol = 2e-4, rtol = 2e-4)) ||
            error("PyTorch reference mismatch")
        push!(errors, maximum(abs.(actual .- expected)))
    end
    inputs = inputs_for(first(cases))
    warmups = 5
    for _ = 1:warmups
        JeffClient.logits(backend, inputs)
    end
    times, gc_times = Float64[], Float64[]
    heap_bytes, gpu_bytes, gpu_allocations = Int[], Int[], Int[]
    for _ = 1:samples
        sample = CUDA.@timed JeffClient.logits(backend, inputs)
        push!(times, 1000sample.time)
        push!(gc_times, 1000sample.cpu_gctime)
        push!(heap_bytes, sample.cpu_bytes)
        push!(gpu_bytes, sample.gpu_bytes)
        push!(gpu_allocations, sample.gpu_memstats.alloc_count)
    end
    ext = Base.get_extension(JeffClient, :JeffClientCUDAExt)
    workspace = ext.workspace(backend.backbone.embedding)
    retained =
        sum(a -> length(a)*sizeof(eltype(a)), workspace.slots) +
        sum(
            slots->sum(a->length(a)*sizeof(eltype(a)), slots),
            values(workspace.layer_slots),
        ) +
        2sizeof(Float32)
    record = Dict(
        "implementation"=>"NativeBackend CUDA.jl",
        "checkpoint"=>abspath(checkpoint),
        "reference"=>abspath(reference),
        "julia"=>string(VERSION),
        "cuda_jl"=>string(pkgversion(CUDA)),
        "cuda_runtime"=>string(CUDA.runtime_version()),
        "cublas"=>string(CUDA.CUBLAS.version()),
        "device"=>CUDA.name(CUDA.device()),
        "device_index"=>device,
        "precision"=>"Float32",
        "math_mode"=>string(CUDA.math_mode()),
        "math_precision"=>string(CUDA.math_precision()),
        "batch"=>size(inputs["input_ids"], 1),
        "sequence_length"=>size(inputs["input_ids"], 2),
        "trim_padding"=>get(ENV, "QDC_CUDA_TRIM_PADDING", "0") == "1",
        "computed_sequence_lengths"=>[
            size(inputs["input_ids"], 2)-QwenDecisionCore.native_sequence_start(
                backend.backbone.embedding,
                inputs["attention_mask"],
                row,
            )+1 for row in axes(inputs["input_ids"], 1)
        ],
        "active_tokens"=>sum(inputs["attention_mask"]),
        "validated_cases"=>length(cases),
        "validated_options"=>size(backend.readout, 2),
        "max_logit_errors"=>errors,
        "samples"=>samples,
        "warmups"=>warmups,
        "median_ms"=>median(times),
        "p95_ms"=>quantile(times, 0.95),
        "times_ms"=>times,
        "gc_times_ms"=>gc_times,
        "heap_bytes"=>heap_bytes,
        "gpu_allocated_bytes"=>gpu_bytes,
        "gpu_allocation_count"=>gpu_allocations,
        "workspace_retained_bytes"=>retained,
        "load_seconds"=>load_seconds,
        "scope"=>"prepared CPU inputs, upload, full sequence, readout, CPU logits return and synchronization; excludes loading, compilation, tokenization, answer calibration",
    )
    open(output, "w") do io
        JSON.print(io, record, 2)
        println(io)
    end
    println(
        "median=",
        record["median_ms"],
        " ms, p95=",
        record["p95_ms"],
        " ms, heap=",
        median(heap_bytes),
        " bytes, GPU allocations=",
        median(gpu_allocations),
        ", GPU bytes=",
        median(gpu_bytes),
    )
    CUDA.pool_status()
end
main()
