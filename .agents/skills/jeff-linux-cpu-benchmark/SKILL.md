---
name: jeff-linux-cpu-benchmark
description: Create or maintain tools/linux-cpu.sh in JeffClient.jl, measure a fair Python (CPU) versus Julia (CPU) comparison on Linux, and update docs/src/performance.md from validated measurements. Use for the Linux CPU benchmark driver and its performance report.
---

# Jeff Linux CPU benchmark

Work from the JeffClient.jl repository root. Deliver an executable
`tools/linux-cpu.sh`, reproducible measured results, and a concise
`docs/src/performance.md` based on those results. When invoked only to create or
edit this skill, make the skill changes without starting the benchmark workflow.

## Repository references

Follow the repository's `AGENTS.md`; consult `memories/MEMORY.md` for CPU and
Python compatibility findings before implementation. Paths below are relative
to the repository root, not this skill directory.

- `tools/mac-M-series.sh`: shell CLI, fresh-process sequencing, metadata capture,
  failure handling, and result directory conventions. Adapt the structure for
  Linux CPU; omit macOS, GPU, and MLX operations.
- `tools/benchmark_inference.jl` and `tools/benchmark_original.jl`: native Julia
  and PythonCall-based PyTorch benchmarks. Reuse their model/input preparation.
- `tools/benchmark_cpu_single_{julia,python}.jl`: full-sequence, single-thread
  adapters. The Julia adapter calls AppleAccelerate and cannot run unchanged on
  Linux. Inspect the current adapters before reusing their logic.
- `tools/mac_benchmark_report.jl`: example of rejecting incomplete or mismatched
  runs before producing a summary.
- `docs/src/profiling.md`: detailed investigations and historical measurements;
  preserve unrelated Apple Silicon results when updating Linux documentation.

## Establish equal work

Choose the thread budget explicitly and record the actual runtime settings.
Offer a strict one-thread comparison. For a multicore comparison, give both
processes the same CPU affinity and physical-core budget; record Julia worker,
BLAS, PyTorch intra-op, and inter-op counts separately. Julia workers plus a
multithreaded BLAS can oversubscribe that budget. Set threading before warm-up;
disable Julia's extra interactive pool where supported. Respect the host's
allowed CPU set and avoid selecting sibling logical CPUs as distinct physical
cores. Affinity does not establish exclusive ownership of those cores.

Use identical checkpoint/revision, Float32 weights and computation, prepared
input, case, batch size, sequence length, padding/mask, and readout. Verify dtype
for the whole Python model. Run both backbones over the full prepared sequence.
Julia's `--python-reference` profile disables trimming, final-query-only and
final-token-only work, recurrent Delta, and Octavian, and uses chunk size 64.
Verify recorded settings; do not equate a cropped Julia run with padded Python.
Include readout and CPU score return in both timed forwards. Different CPU
kernels/BLAS libraries may remain; disclose them rather than calling this a
comparison of language overhead alone.

The parcel input has fewer active choices than the trained readout. Check every
readout logit, including inactive choices, against an independent PyTorch
reference before accepting a run. Record output shape, maximum error, numerical
tolerances, and validation success. Do not validate only the question's active
options or replace nonfinite values to make a comparison pass.

On Linux, installed FLA may select GPU-only Triton for CPU execution. If this
occurs, use process-local `inspect.unwrap` selection of the original
Transformers `torch_chunk_gated_delta_rule` and
`torch_recurrent_gated_delta_rule`; assert their origin is the Transformers
module. Set `USE_HUB_KERNELS=NO` when needed and record the adaptation. Leave
installed Python sources unchanged. Python model/library use stays through
PythonCall.jl, as required by the repository.

## Implement and run the driver

For inference-kernel optimization or allocation/GC investigations, follow
[jeff-cpu-performance](../jeff-cpu-performance/SKILL.md), then return here for
the matched comparison. Benchmark tooling alone is not evidence of a speedup.

Create or update `tools/linux-cpu.sh` with `--help` and configurable checkpoint,
reference, output directory, samples, fresh-process repeats, warm-ups, and
thread budget/affinity. Follow existing ignored `artifacts/benchmarks/` output
conventions. Validate arguments and dependencies before measurement; reject
output directories containing prior results. Keep dependency setup outside
timed runs. Make environment preparation explicit and reproducible.

Run implementations sequentially in fresh processes, with equal warm-up and
sample counts. Prefer at least 10 warm-up forwards and 30 measured forwards,
with at least two fresh runs; a smoke test is not a published benchmark.
Alternate Python/Julia order across repeats. Exclude imports, checkpoint
loading, compilation, tokenization, and reference validation from warm timing.
Align GC policy and use one forward per sample. Do not run profilers, tests, or
other benchmark jobs concurrently with the measurements.

Save per-forward timings and per-run JSON/logs, CPU topology/affinity,
Julia/Python/package and BLAS versions, checkpoint and input hashes, source
revision and dirty-tree state, relevant source hashes, actual thread counts,
and computation/validation settings. Fail the driver if either process fails.
Generate a summary only after checking sample counts, shapes, dtypes, full
sequence length, thread budget, numerical guards, and matching input/model
identity across all required runs. Preserve failed logs without presenting
partial results as a successful comparison.

Check shell syntax and exercise `--help`, invalid arguments, a small end-to-end
run, and rejection of incomplete/mismatched results before full measurement.
Format changed Julia files and run the relevant checks specified by the
repository. Changes to inference algorithms are separate optimization work;
do not silently enable them merely to improve a benchmark ratio.

## Publish the evidence

Update `docs/src/performance.md` with the exact driver invocation, hardware and
software conditions, excluded work, validation results, and a compact table of
each fresh run's median and p95. State the ratio direction explicitly:
`Python median / Julia median` is Julia's speedup. Show variability and avoid
selecting only the fastest repeat. Distinguish strict single-thread from equal
multicore-budget results and separate results using different BLAS libraries.

Link a compact validated JSON under `docs/src/assets/benchmarks/`; keep models
and large raw logs in ignored artifacts. Distinguish Julia heap allocation,
retained model memory, and process peak RSS when reporting memory. Keep
performance.md focused on current reproducible results; link profiling.md for
investigations and history, and update memories/MEMORY.md with new reusable
findings rather than copying measurements into this skill.

If the host or reference cannot execute, report the concrete failure and leave
the documentation clear about what remains unmeasured. Never fabricate timings
or substitute older measurements as results of the new driver. Creating this
skill does not imply running it, committing, or pushing; perform those actions
when included in the user's task.
