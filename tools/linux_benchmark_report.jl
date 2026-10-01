using JSON
using Statistics

include("linux_benchmark_config.jl")
length(ARGS) == 4 ||
    error("Usage: linux_benchmark_report.jl OUTPUT SAMPLES REPEATS WARMUPS")
output = abspath(ARGS[1])
samples, repeats, warmups = parse.(Int, ARGS[2:4])
config = JSON.parsefile(joinpath(output, "config.json"))
reference = JSON.parsefile(config["reference"])
cases = reference isa AbstractDict ? reference["cases"] : reference
expected = first(cases)
expected_shape = [length(expected["logits"]), length(first(expected["logits"]))]
results = Dict{String,Any}()
for implementation in ("python", "julia")
    records = []
    for run = 1:repeats
        path = joinpath(output, "$implementation-run$run.json")
        isfile(path) || error("Missing completed run: $path")
        r = JSON.parsefile(path)
        r["samples"] == length(r["times_ms"]) == samples ||
            error("Sample count differs: $path")
        r["warmup_forwards"] == warmups || error("Warm-ups differ: $path")
        all(x -> isfinite(x) && x > 0, r["times_ms"]) || error("Invalid timings: $path")
        isapprox(median(r["times_ms"]), r["median_ms"]; rtol = 1e-10) ||
            error("Median differs: $path")
        r["validation_passed"] === true || error("Invalid scores: $path")
        r["validated_options"] == expected_shape[2] ||
            error("Partial output validation: $path")
        r["output_shape"] == expected_shape || error("Output shape differs: $path")
        r["batch_size"] == length(expected["inputs"]["input_ids"]) ||
            error("Batch differs: $path")
        r["sequence_length"] == length(first(expected["inputs"]["input_ids"])) ||
            error("Sequence differs: $path")
        r["active_tokens"] == sum.(expected["inputs"]["attention_mask"]) ||
            error("Mask differs: $path")
        r["reference_sha256"] == config["reference_sha256"] ||
            error("Input identity differs: $path")
        r["checkpoint"] == config["checkpoint"] ||
            error("Checkpoint identity differs: $path")
        r["thread_budget"] == config["threads"] || error("Budget differs: $path")
        expand_cpus(r["cpu_affinity"]) == config["cpus"] || error("Affinity differs: $path")
        if implementation == "python"
            r["cpu_threads"] == config["threads"] && r["interop_threads"] == 1 ||
                error("Python threads differ")
            r["parameter_dtypes"] == ["torch.float32"] ||
                error("Mixed/non-Float32 parameters")
        else
            r["julia_worker_threads"] == config["threads"] && r["blas_threads"] == 1 ||
                error("Julia threads differ")
            r["weight_dtype"] == "Float32" || error("Julia precision differs")
            all(==(r["sequence_length"]), r["cpu_computed_sequence_lengths"]) ||
                error("Trimmed computation")
            for key in (
                "cpu_trim_padding_enabled",
                "cpu_final_query_enabled",
                "cpu_final_token_only_enabled",
                "cpu_recurrent_delta_enabled",
                "cpu_octavian_delta_enabled",
            )
                r[key] === false || error("Unequal computation: $key")
            end
            r["cpu_delta_chunk_size"] == 64 || error("Chunk size differs")
        end
        push!(records, r)
    end
    results[implementation] = records
end
summary = Dict(
    "config" => config,
    "samples_per_run" => samples,
    "fresh_processes_per_implementation" => repeats,
    "warmups_per_run" => warmups,
    "hardware" => read(joinpath(output, "hardware.txt"), String),
    "results" => results,
    "conditions" => "Float32/full sequences/readout/all output validation/CPU scores; sequential fresh processes, alternating order; equal affinity and physical-core budget; host not exclusively isolated.",
)
write(joinpath(output, "summary.json"), JSON.json(summary, 2) * "\n")
open(joinpath(output, "summary.md"), "w") do io
    println(io, "# Linux matched CPU comparison\n\n", summary["conditions"], "\n")
    println(io, "| Implementation | Run | Median (ms) | p95 (ms) | Max logit error |")
    println(io, "|---|---:|---:|---:|---:|")
    for run = 1:repeats, implementation in ("python", "julia")
        r = results[implementation][run]
        println(
            io,
            "| $implementation | $run | ",
            round(r["median_ms"]; digits = 3),
            " | ",
            round(r["p95_ms"]; digits = 3),
            " | ",
            r["max_logit_error"],
            " |",
        )
    end
    println(io, "\nJulia speedup = Python median / Julia median:")
    for run = 1:repeats
        println(
            io,
            "\n- Run $run: ",
            round(
                results["python"][run]["median_ms"] / results["julia"][run]["median_ms"];
                digits = 4,
            ),
            "×",
        )
    end
end
println("Wrote validated summary.json and summary.md to ", output)
