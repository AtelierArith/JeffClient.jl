# Warm, synchronized forward timing of NativeBackend on CPU, Metal or CUDA.
# Each sample covers input upload, the full forward, readout and the CPU logits
# return; checkpoint loading, compilation and tokenization are excluded.
#
#   julia --threads=8 --project=tools tools/benchmark.jl DEVICE [options]
#
# DEVICE is cpu, metal or cuda. Options:
#   --checkpoint DIR   local checkpoint (default: the pinned Hub revision)
#   --reference FILE   prepared inputs and PyTorch logits
#                      (default: examples/data/parcel_reference.json)
#   --case N           reference case to time (default: 1)
#   --samples N        timed forwards (default: 30)
#   --warmups N        untimed forwards before timing (default: 5)
#   --gpu N            CUDA device index (default: 0)
#   --accelerate       Apple silicon CPU only: load AppleAccelerate and forward
#                      BLAS to Accelerate before the model loads (opt-in fast
#                      path; without it the default BLAS is used)
#   --output FILE      also write the JSON record to FILE
const USAGE = "Usage: benchmark.jl cpu|metal|cuda [--checkpoint DIR] [--reference FILE] [--case N] [--samples N] [--warmups N] [--gpu N] [--accelerate] [--output FILE]"

const ACCELERATE = "--accelerate" in ARGS
const CLI = [arg for arg in ARGS if arg != "--accelerate"]
const DEVICE = isempty(CLI) ? error(USAGE) : Symbol(CLI[1])
DEVICE in (:cpu, :metal, :cuda) || error(USAGE)
DEVICE == :metal && import Metal
DEVICE == :cuda && import CUDA
if ACCELERATE
    Sys.isapple() || error("--accelerate is only available on macOS")
    import AppleAccelerate
end

using JeffClient, LinearAlgebra, Statistics
import QwenDecisionCore
const JSON = QwenDecisionCore.JSON

function options(args)
    values = Dict(
        "checkpoint" => "",
        "reference" =>
            joinpath(@__DIR__, "..", "examples", "data", "parcel_reference.json"),
        "case" => "1",
        "samples" => "30",
        "warmups" => "5",
        "gpu" => "0",
        "output" => "",
    )
    iseven(length(args)) || error(USAGE)
    for (flag, value) in Iterators.partition(args, 2)
        key = chopprefix(flag, "--")
        haskey(values, key) && startswith(flag, "--") || error(USAGE)
        values[key] = value
    end
    return values
end

matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])

function synchronized_logits(backend, inputs)
    result = logits(backend, inputs)
    DEVICE == :metal && Metal.synchronize()
    return result
end

# CUDA.@timed must not be expanded unless CUDA is loaded.
if DEVICE == :cuda
    @eval function timed(backend, inputs)
        sample = CUDA.@timed synchronized_logits(backend, inputs)
        return (; sample.time, bytes = sample.cpu_bytes, gpu_bytes = sample.gpu_bytes)
    end
else
    function timed(backend, inputs)
        sample = @timed synchronized_logits(backend, inputs)
        return (; sample.time, sample.bytes, gpu_bytes = missing)
    end
end

function device_name()
    DEVICE == :cuda && return CUDA.name(CUDA.device())
    DEVICE == :metal && return string(Metal.device().name)
    return Sys.cpu_info()[1].model
end

function main(args)
    opts = options(args)
    samples, warmups = parse(Int, opts["samples"]), parse(Int, opts["warmups"])
    samples >= 3 || error("Use at least three samples.")
    if DEVICE == :cuda
        CUDA.device!(parse(Int, opts["gpu"]))
        CUDA.allowscalar(false)
    elseif DEVICE == :metal
        Metal.functional() || error("A functional Apple GPU is required.")
        Metal.allowscalar(false)
    end
    reference = JSON.parsefile(opts["reference"])
    cases = reference isa AbstractDict ? reference["cases"] : reference
    case = cases[parse(Int, opts["case"])]
    inputs = Dict(name => matrix(rows, Int64) for (name, rows) in case["inputs"])
    expected = matrix(case["logits"], Float32)

    checkpoint = opts["checkpoint"]
    if isempty(checkpoint)
        pinned = JSON.parsefile(joinpath(@__DIR__, "..", "examples", "data", "parcel.json"))
        checkpoint = resolve_checkpoint(pinned["model"]; revision = pinned["revision"])
    end
    load_seconds = @elapsed backend = NativeBackend(checkpoint; device = DEVICE)
    first_seconds = @elapsed actual = synchronized_logits(backend, inputs)
    max_error = maximum(abs.(actual .- expected))
    all(abs.(actual .- expected) .<= 2.0f-4 .+ 2.0f-4 .* abs.(expected)) ||
        error("Logits differ from the reference (max error $max_error).")
    for _ = 1:warmups
        synchronized_logits(backend, inputs)
    end
    GC.gc(true)
    trials = [timed(backend, inputs) for _ = 1:samples]
    times = [1000 * t.time for t in trials]

    record = Dict(
        "device" => String(DEVICE),
        "device_name" => device_name(),
        "julia" => string(VERSION),
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "apple_accelerate" => ACCELERATE,
        "precision" => "Float32",
        "reference" => abspath(opts["reference"]),
        "batch" => size(inputs["input_ids"], 1),
        "sequence_length" => size(inputs["input_ids"], 2),
        "active_tokens" => vec(sum(inputs["attention_mask"]; dims = 2)),
        "max_logit_error" => max_error,
        "load_seconds" => load_seconds,
        "first_forward_seconds" => first_seconds,
        "warmups" => warmups,
        "samples" => samples,
        "median_ms" => median(times),
        "p95_ms" => quantile(times, 0.95),
        "times_ms" => times,
        "julia_heap_bytes" => median(t.bytes for t in trials),
    )
    if DEVICE == :cuda
        record["cuda_jl"] = string(pkgversion(CUDA))
        record["cuda_runtime"] = string(CUDA.runtime_version())
        record["cublas"] = string(CUDA.CUBLAS.version())
        record["gpu_allocated_bytes"] = median(t.gpu_bytes for t in trials)
    elseif DEVICE == :metal
        record["metal_jl"] = string(pkgversion(Metal))
    end
    json = JSON.json(record, 2)
    println(json)
    if !isempty(opts["output"])
        mkpath(dirname(abspath(opts["output"])))
        write(opts["output"], json * "\n")
    end
    return record
end

main(CLI[2:end])
