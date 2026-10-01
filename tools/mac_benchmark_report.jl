using JSON
using SHA

length(ARGS) == 3 || error("Usage: mac_benchmark_report.jl OUTPUT_DIR SAMPLES REPEATS")
output = abspath(ARGS[1])
samples, repeats = parse.(Int, ARGS[2:3])
groups = [
    ("cpu-single-python", "CPU 1 thread: Python / PyTorch", :cpu_single),
    ("cpu-single-julia", "CPU 1 thread: Julia / Accelerate", :cpu_single),
    ("gpu-pytorch", "GPU: PyTorch MPS Float32", :gpu),
    ("gpu-metal", "GPU: Julia Metal.jl full-sequence adapter", :gpu),
    ("cpu-eight-python", "CPU: PyTorch 8, additional budget", :automatic),
    (
        "cpu-auto-julia",
        "CPU: Julia 8 / Accelerate automatic, additional budget",
        :automatic,
    ),
    ("gpu-mlx", "GPU: MLX reference-normalization adapter", :mlx),
    ("cpu-mlx", "CPU: MLX framework-managed, reference-normalization adapter", :mlx),
]
results = Dict{String,Any}()
reference_shape = nothing
for (prefix, label, kind) in groups
    files = [joinpath(output, "$prefix-run$i.json") for i = 1:repeats]
    any(isfile, files) || continue
    all(isfile, files) || error("Incomplete group: $prefix")
    records = JSON.parsefile.(files)
    for record in records
        record["samples"] == samples || error("Sample count differs: $prefix")
        shape = (record["batch_size"], record["sequence_length"], record["active_tokens"])
        if reference_shape === nothing
            global reference_shape = shape
        else
            shape == reference_shape || error("Input shapes/masks differ: $prefix")
        end
        get(record, "validation_passed", true) || error("Invalid scores: $prefix")
        if prefix == "cpu-single-python"
            record["cpu_threads"] == record["interop_threads"] == 1 ||
                error("Python thread count differs")
            record["weight_dtype"] == "torch.float32" || error("Python precision differs")
        elseif prefix == "cpu-single-julia"
            record["julia_worker_threads"] ==
            record["blas_threads"] ==
            record["accelerate_threads"] ==
            1 || error("Julia thread count differs")
        elseif prefix == "gpu-pytorch"
            record["weight_dtype"] == "torch.float32" || error("MPS precision differs")
            record["cpu_input_upload_included"] || error("MPS upload excluded")
        elseif kind == :mlx
            record["reference_norm"] || error("Unadapted MLX cannot enter this comparison")
            record["weight_dtypes"] == ["mlx.core.float32"] ||
                error("MLX precision differs")
        end
        for key in ("cpu_computed_sequence_lengths", "metal_computed_sequence_lengths")
            haskey(record, key) || continue
            all(==(record["sequence_length"]), record[key]) ||
                error("Padding was trimmed: $prefix")
        end
    end
    results[prefix] = Dict("label" => label, "category" => string(kind), "runs" => records)
end
all(
    haskey(results, key) for
    key in ("cpu-single-python", "cpu-single-julia", "gpu-pytorch", "gpu-metal")
) || error("Missing core comparison")
root = abspath(joinpath(@__DIR__, ".."))
source_files = [joinpath(@__DIR__, "mac-M-series.sh")]
append!(
    source_files,
    [
        joinpath(@__DIR__, name) for name in readdir(@__DIR__) if
        (startswith(name, "benchmark") || startswith(name, "mac_benchmark")) &&
            endswith(name, ".jl")
    ],
)
hashes = Dict(relpath(file, root) => bytes2hex(open(sha256, file)) for file in source_files)
summary = Dict(
    "julia_version" => string(VERSION),
    "machine" => Sys.MACHINE,
    "cpu" => Sys.cpu_info()[1].model,
    "source_hashes" => hashes,
    "runtime" => JSON.parsefile(joinpath(output, "runtime.json")),
    "source_commit" => strip(read(joinpath(output, "source-commit.txt"), String)),
    "hardware" => read(joinpath(output, "hardware.txt"), String),
    "model_reference_and_tool_hashes" => read(joinpath(output, "hashes.txt"), String),
    "samples_per_run" => samples,
    "fresh_processes_per_implementation" => repeats,
    "results" => results,
    "conditions" => "Full sequence / Float32 / readout / CPU scores / GPU completion; GPU CPU-input upload included. One-thread CPU primary; automatic CPU and MLX CPU have different thread budgets. Sequential processes; host not isolated or affinity-pinned.",
)
write(joinpath(output, "summary.json"), JSON.json(summary, 2) * "\n")
open(joinpath(output, "summary.md"), "w") do io
    println(io, "# Apple Silicon matched inference comparison\n")
    println(io, summary["conditions"], "\n")
    for kind in (:cpu_single, :automatic, :gpu, :mlx)
        any(group -> group[3] == kind && haskey(results, group[1]), groups) || continue
        println(io, "## ", kind, "\n")
        println(io, "| Implementation | Run | Median (ms) | p95 (ms) | Samples |")
        println(io, "|---|---:|---:|---:|---:|")
        for (prefix, label, category) in groups
            category == kind && haskey(results, prefix) || continue
            for (index, record) in enumerate(results[prefix]["runs"])
                println(
                    io,
                    "| ",
                    label,
                    " | ",
                    index,
                    " | ",
                    round(record["median_ms"]; digits = 3),
                    " | ",
                    round(record["p95_ms"]; digits = 3),
                    " | ",
                    record["samples"],
                    " |",
                )
            end
        end
        println(io)
    end
end
println("Wrote validated summary.json and summary.md to ", output)
