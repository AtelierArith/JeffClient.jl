# Linux-only comparison adapter. All model/library access is through Julia.
using JSON
using SHA

Sys.islinux() || error("This adapter requires Linux.")
length(ARGS) == 7 || error(
    "Usage: benchmark_linux_cpu.jl python|julia CHECKPOINT REFERENCE SAMPLES OUTPUT THREADS WARMUPS",
)
implementation, checkpoint, reference, samples, output, threads, warmups = ARGS
budget, warmup_count = parse.(Int, [threads, warmups])
budget >= 1 && warmup_count >= 2 || error("Invalid thread budget/warmups.")

# Fail loudly if upstream benchmark changes invalidate an adapter.
function adapt(source, old, new)
    length(findall(old, source)) == 1 || error("Benchmark adapter no longer matches: $old")
    return replace(source, old => new)
end

if implementation == "python"
    ENV["USE_HUB_KERNELS"] = "NO"
    include("python_env.jl")
    using PythonCall
    pyexec(
        raw"""
import inspect
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
using QwenDecisionCore
torch.set_num_interop_threads(1)
for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
    original = inspect.unwrap(getattr(qwen, name))
    assert original.__module__ == qwen.__name__, (name, original.__module__)
    setattr(qwen, name, original)
""",
        pydict(),
    )
    source = read(joinpath(@__DIR__, "benchmark_original.jl"), String)
    source = adapt(source, "cpu_threads=8", "cpu_threads=$budget")
    source = adapt(
        source,
        "batch = PreparedBatch",
        "count = model.readout.out_features\nbatch = PreparedBatch",
    )
    source = adapt(
        source,
        "    for _ in range(warmups - 1):",
        "    import gc\n    gc.collect()\n    for _ in range(warmups):",
    )
    source = adapt(
        source,
        "\"samples\": len(times),",
        "\"times_ms\": times, \"validated_options\": count, \"output_shape\": list(actual.shape), \"samples\": len(times),",
    )
    empty!(ARGS)
    append!(ARGS, [checkpoint, "cpu", reference, samples, output, "--warmups=$warmups"])
    include_string(Main, source, joinpath(@__DIR__, "benchmark_original.jl"))
    record = JSON.parsefile(output)
    record["interop_threads"] = pyconvert(Int, pyimport("torch").get_num_interop_threads())
    record["torch_parallel_info"] =
        pyconvert(String, pyimport("torch").__config__.parallel_info())
    record["cpu_reference_adaptation"] = "Original Transformers torch Delta functions; USE_HUB_KERNELS=NO"
    record["max_logit_error"] = record["max_active_logit_error"]
elseif implementation == "julia"
    # Optional BLAS environment, prepared separately, never installed in a timed process.
    if haskey(ENV, "JEFF_BENCH_BLAS_PROJECT")
        push!(LOAD_PATH, abspath(ENV["JEFF_BENCH_BLAS_PROJECT"]))
        @eval using MKL
        Base.invokelatest(() -> MKL.set_num_threads(1))
    end
    source = read(joinpath(@__DIR__, "benchmark_inference.jl"), String)
    source = adapt(source, "main()\n", "")
    source = adapt(
        source,
        "        for _ = 2:warmups",
        "        GC.gc(true)\n        for _ = 1:warmups",
    )
    source = adapt(source, "        GC.gc(true)\n        trial", "        trial")
    source = adapt(
        source,
        "samples=samples evals=1 seconds=120",
        "samples=samples evals=1 seconds=3600 gctrial=false gcsample=false",
    )
    source = adapt(
        source,
        "\"samples\" => length(trial.times),",
        "\"times_ms\" => trial.times ./ 1e6, \"gc_times_ms\" => trial.gctimes ./ 1e6, \"samples\" => length(trial.times),",
    )
    empty!(ARGS)
    include_string(Main, source, joinpath(@__DIR__, "benchmark_inference.jl"))
    record = Base.invokelatest() do
        Threads.nthreads(:default) == budget || error("Julia thread budget differs.")
        QwenDecisionCore.initialize_cpu!()
        BLAS.set_num_threads(1)
        QwenDecisionCore.with_cpu_settings(
            :trim_padding => false,
            :final_query => false,
            :final_token_only => false,
            :recurrent_delta => false,
            :octavian_delta => false,
            :delta_chunk_size => 64,
        ) do
            run_benchmark(
                [checkpoint, "cpu", reference, "1", samples, output],
                :linux_equal_work,
                warmup_count,
            )
        end
        JSON.parsefile(output)
    end
    record["weight_dtype"] = "Float32"
    cases = JSON.parsefile(reference)
    cases = cases isa AbstractDict ? cases["cases"] : cases
    record["validated_options"] = length(first(first(cases)["logits"]))
    record["output_shape"] = [record["batch_size"], record["validated_options"]]
else
    error("Choose python or julia.")
end

record["validation_passed"] = true
record["validation_atol"] = record["validation_rtol"] = 2e-4
record["cpu_affinity"] =
    match(r"Cpus_allowed_list:\s*([^\n]+)", read("/proc/self/status", String))[1]
record["checkpoint"] = realpath(checkpoint)
record["reference_sha256"] = bytes2hex(open(sha256, reference))
record["thread_budget"] = budget
record["timing_gc_policy"] = "Full collection after cold validation, before warm-up; automatic GC enabled; no per-sample collection"
write(output, JSON.json(record, 2) * "\n")
