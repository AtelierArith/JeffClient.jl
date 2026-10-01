---
name: jeff-cpu-performance
description: Investigate and optimize JeffClient.jl native Julia CPU inference using time and allocation profiles, owned workspaces and numerical/type checks, then verify speed against Python under matched conditions. Use for CPU bottlenecks, GC latency and repeated allocation reduction work.
---

# Jeff CPU performance

Run from the JeffClient.jl repository root. Follow `AGENTS.md` and read the CPU
findings in `memories/MEMORY.md` before choosing an experiment. This skill
contains the reusable investigation loop; keep measured results and
library-specific findings in the repository memory and documentation.

## Establish a trustworthy baseline

For Python comparisons, use the protocol and driver in
[jeff-linux-cpu-benchmark](../jeff-linux-cpu-benchmark/SKILL.md).
Record the actual policy, BLAS library/thread counts, source hashes and input
identity for every baseline and candidate. Full-sequence reference mode must
remain full-sequence through final normalization and readout. A different
amount of work, BLAS configuration, or warm-up protocol cannot establish the
effect of a kernel change.

Keep a before-change benchmark and an isolated time/allocation profile:

- `tools/profile_native_cpu.jl` supplies `@code_warntype`, JET, `Profile` and
  sampled `Profile.Allocs` output. Use `with_cpu_settings` to match the baseline
  profile; for a Python-reference investigation disable trim, final query,
  final token and recurrent Delta, and set chunk size 64 and Octavian off.
- `tools/time_cpu_phases.jl` can split stages. Inspect its current implementation
  first: a diagnostic that allocates generic RMS/attention outputs does not
  represent a newer workspace path. Update it when necessary before trusting
  its stage allocation or timing breakdown.
- `tools/linux-cpu.sh` supplies the publishable warmed comparison. The profile
  tool's short warm-up diagnostic is not a substitute for that benchmark.

Run timed trials and profiles separately from tests, compilation jobs and
other benchmarks. Preserve logs under ignored `artifacts/`; do not restart a
still-live process simply because it has not printed output yet.

## Distinguish computation, allocation and GC

Rank both sampled time sites and allocation sites before editing. Sampled
allocation totals are not exact heap totals. Type stability does not imply
allocation-free execution: check the measured allocation estimate separately.
Record the estimator the harness actually provides rather than inventing
per-sample allocation statistics.

The Linux Julia adapter records `times_ms` and `gc_times_ms`. Pair them by
sample to test whether slow forwards coincide with GC. Large allocation
alone does not prove GC causes the latency gap; a slow sample with little GC
requires another explanation. GC-subtracted timings are diagnostic and must
not replace end-to-end inference timings in the speed comparison.

## Make one bounded kernel/workspace experiment

Locate the hot path through CodeGraph when available, following `AGENTS.md`.
CPU specializations live mainly in `src/native_cpu.jl`, with portable SIMD
hooks in `ext/JeffClientLoopVectorizationExt.jl`. Preserve generic/GPU dispatch
and the Apple Accelerate policy when changing a portable CPU path.

Prefer reuse where arrays are created: project with `mul!` into existing
storage, overwrite owned normalization outputs, and reuse scratch across
layers after all consumers finish. Keep separate storage for values still
needed by the same layer. Prove that an input or model weight cannot alias a
disposable buffer before mutating it. An internal function returning borrowed
scratch must be consumed before that scratch is reused; returned public
scores must remain valid after subsequent calls and GC.

For parallel heads, give each worker exclusive score/value/state buffers and
wait for workers before reuse. Keep CPU policy propagation explicit where a
worker reads task-local settings. Reset recurrent state and overwrite input-
dependent masks on every use, including a different mask at the same length.
Compute immutable positional tables once per forward when their reuse is
valid. Start with forward-owned storage; cross-request pools require separate
evidence for checkout ownership, concurrency, retained memory and eviction.

Keep SIMD guards meaningful: preserve NaN/Inf, signed zero, subnormals and
out-of-domain scalar behavior. Do not zero unsafe values merely to make a
numerical guard pass. Shared NamedTuple factories should preserve correlated
field types; check JET for newly introduced runtime dispatch rather than
assuming a concrete outer return type proves the whole path is stable.

## Validate, measure and decide

Format changed Julia files. Run checks suited to the changed path:

- `test/runtests.jl` covers independent PyTorch fixtures and public inference.
- `tools/verify_cpu_vector_math.jl` covers exceptional SIMD lanes and native
  ownership tests; run one-worker and multiple-worker cases for parallel edits.
- Add focused tests for changed buffer ownership: poisoned scratch, changed
  length, same-length changed mask, GC and concurrent calls. Validate the real
  model's entire readout as well; a fixture alone does not establish real-model
  numerical accuracy or speed.
- Re-run type checks for the changed path. Keep exact measured allocation and
  an isolated profile separate from the timing comparison.

Repeat the baseline/candidate timing without concurrent diagnostic work, then
run fresh Python/Julia comparisons with alternating order. Report median,
p95, GC behavior and allocation independently. A memory reduction with flat
latency is a memory improvement, not a speed win. Do not adopt a more complex
fusion based on an improvement smaller than the observed variability; retain
its evidence in ignored artifacts if it is rejected.

Finish the user's actual performance target only after fresh matched results
and numerical/type checks prove it. Update `docs/src/performance.md` with
current reproducible results, link detailed investigations from
`docs/src/profiling.md`, and update `memories/MEMORY.md`. Preserve historical
results as history rather than relabeling them as a new measurement. Reconcile
this skill when repeated work reveals a reusable procedure; do not copy run
numbers or dependency-version findings into it.
