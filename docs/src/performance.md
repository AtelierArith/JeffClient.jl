# Measured inference speed

## Current source CPU benchmarks (2026-10-01)

Measured commit [`5f2d0e9`](https://github.com/AtelierArith/JeffClient.jl/commit/5f2d0e9e963b9a812bf561eeb37202b7334e513a)
on **Apple M4 / arm64 macOS / Julia 1.13.1 / Float32**, with **8 Julia workers**
and **30 warmed forwards per configuration**. The Intel trials and older M4
measurements below remain historical results; compare only matched hardware
and settings.

All rows use **Jeff's actual trained 0.8B weights**, checkpoint
`mstrasser/Jeff-Qwen3.5-0.8B`, pinned to
`0f212b3e72acb4dde3f7da61e925d6ab7f819990`. The input is case 1 of
`examples/data/parcel_reference.json`: B1, logical length 256, 101 active tokens.
No ONNX or tiny test graph is involved. Timing includes `logits`, the trained
readout and scores on CPU. Loading, download, tokenization and first-call
compilation are excluded. Each configuration ran in a separate process,
sequentially, with inherited tuning overrides cleared.

| Configuration | Computed tokens | Median | p95 | Julia heap per forward | Allocations |
|---|---:|---:|---:|---:|---:|
| Default / OpenBLAS 8 | 256 | 1503.13 ms | 1545.25 ms | 916.4 MiB | 22,016 |
| README / Accelerate | 101 | 183.34 ms | 189.79 ms | 367.6 MiB | 22,707 |
| Portable / projection workspace off | 101 | 351.98 ms | 360.79 ms | 253.3 MiB | 12,601 |
| Portable / projection workspace on | 101 | 359.10 ms | 363.93 ms | 104.0 MiB | 11,895 |
| Accelerate + latest opt-ins | 101 | 205.76 ms | 213.28 ms | 121.0 MiB | 4,380 |

**The README configuration was fastest among these measured settings.**
Its advantage over the default includes skipping padding and changing the
math library, so the rows do not perform equal arithmetic work.

The new Delta projection workspace reduced portable-path cumulative heap
allocation by **58.9%**, but did not improve its median latency on this M4
(351.98 → 359.10 ms). These are single 30-call trials, not proof of a general
regression or improvement. The Intel gain reported below does not establish
an M4 gain. Accelerate with the latest opt-ins likewise allocates less than
the README setting but was slower. Heap allocation is not peak resident
memory, live workspace size or native BLAS scratch memory.

All five runs passed the saved independent PyTorch logit guard; maximum
absolute errors were 1.05e-5–1.24e-5. This checks one prepared input, not
classification accuracy across a dataset. Exact results, flag values, model
load/first-call times, retained model memory and process peak RSS are preserved
in the [raw benchmark JSON](assets/benchmarks/cpu-2026-10-01-5f2d0e9.json).
Python was not rebenchmarked in this refresh; no new Python speed ratio is
claimed.

Configuration details:

- **Default:** optional flags off, OpenBLAS 8 threads, no padding trim.
- **README:** Accelerate 0.7.0, vector math, parallel Delta heads, MLP workspace
  and trim. Accelerate reports 10 framework-managed threads; LBT's reported
  count of 8 does not represent its actual worker count.
- **Portable:** OpenBLAS 1 thread, LoopVectorization 0.12.174 portable vector
  math and blockwise SiLU, Delta normalization loop, parallel projections,
  projection thread scope, parallel full-attention heads, recurrent DeltaNet,
  parallel Delta heads, MLP/Delta workspace, trim and final query. The two rows
  differ only in `JEFF_CPU_DELTA_PROJECTION_WORKSPACE`. Domain guards retain
  scalar fallback where needed.
- **Accelerate + latest opt-ins:** README plus Delta workspace, recurrent
  DeltaNet, final query, Delta normalization loop and Delta projection
  workspace. Portable vector math and parallel projections/full heads are off.

For context, the preceding M4 refresh on `4dd4a40` measured default 1,491.94 ms
and README 183.05 ms. The current 1,503.13 / 183.34 ms medians show no clear
speed gain for those unchanged configurations; no statistical significance is
claimed. [Previous raw results](assets/benchmarks/cpu-2026-10-01.json) retain
that snapshot's source hash and settings.

### Reproduce these measurements

Use a fresh shell without other `JEFF_CPU_*` overrides. The tools environment
uses this checkout's source. Resolve a stale ignored Manifest before running:

```sh
julia --project=tools -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(; workspace=true)'
CHECKPOINT=$(julia --project=tools -e 'using JeffClient; print(resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B"; revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990"))')

# Default.
JEFF_BLAS_THREADS=8 julia --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/cpu-current-5f2d0e9/default.json

# README / Accelerate.
JEFF_CPU_ACCELERATE=1 JEFF_CPU_VECTOR_MATH=1 JEFF_CPU_PARALLEL_HEADS=1 \
JEFF_CPU_MLP_WORKSPACE=1 JEFF_CPU_TRIM_PADDING=1 \
julia --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/cpu-current-5f2d0e9/readme-accelerate.json

# Portable path: matched comparison of projection workspace off/on.
for projection_workspace in 0 1; do
  JEFF_BLAS_THREADS=1 JEFF_CPU_PORTABLE_VECTOR_MATH=1 JEFF_CPU_VECTOR_MATH_BLOCKS=1 \
  JEFF_CPU_DELTA_NORM_LOOP=1 JEFF_CPU_PARALLEL_PROJECTIONS=1 \
  JEFF_CPU_PROJECTION_THREAD_SCOPE=1 JEFF_CPU_PARALLEL_FULL_HEADS=1 \
  JEFF_CPU_RECURRENT_DELTA=1 JEFF_CPU_PARALLEL_HEADS=1 JEFF_CPU_MLP_WORKSPACE=1 \
  JEFF_CPU_DELTA_WORKSPACE=1 JEFF_CPU_TRIM_PADDING=1 JEFF_CPU_FINAL_QUERY=1 \
  JEFF_CPU_DELTA_PROJECTION_WORKSPACE="$projection_workspace" \
  julia --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 "artifacts/cpu-current-5f2d0e9/portable-${projection_workspace}.json"
done

# Accelerate plus latest opt-ins.
JEFF_CPU_ACCELERATE=1 JEFF_CPU_VECTOR_MATH=1 JEFF_CPU_PARALLEL_HEADS=1 \
JEFF_CPU_MLP_WORKSPACE=1 JEFF_CPU_TRIM_PADDING=1 JEFF_CPU_DELTA_WORKSPACE=1 \
JEFF_CPU_FINAL_QUERY=1 JEFF_CPU_RECURRENT_DELTA=1 JEFF_CPU_DELTA_NORM_LOOP=1 \
JEFF_CPU_DELTA_PROJECTION_WORKSPACE=1 \
julia --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/cpu-current-5f2d0e9/accelerate-projection-workspace.json
```

## Intel CPU optimization trial (2026-10-01)

These results are separate from the Apple M4 measurements below. On an Intel
Core i9-9900K, Julia 1.13.1, Float32, Julia/BLAS 8 threads, Apple Accelerate
0.7.0, the same pinned 0.8B parcel input (B1, padded length 256, active 101)
measured 495.25 ms median / 511.03 ms p95 before these changes (20 warm calls).
MKL is not used. The comparison baseline already enables parallel heads,
MLP workspace, vector math, and leading-padding trim.

CPU convolution weights are now channel-contiguous at load time. Optional
`JEFF_CPU_DELTA_WORKSPACE=1` reuses worker-owned state/full/tail buffers across
layers within one forward, with a reset per head and no shared persistent
cache. Optional `JEFF_CPU_FINAL_QUERY=1` computes only the last query in the
last full-attention layer; keys and values still read the entire context.
Together these measured 417.53 ms median / 441.45 ms p95, 310,814,640 cumulative
Julia heap bytes / 13,838 allocations, versus 392,752,624 bytes / 22,671
allocations initially. Maximum logit error versus the saved independent
reference was `1.05e-5`. This is **1.19×, not the requested 2×**; the target is
at most 247.63 ms under the same conditions. Heap bytes do not measure peak
resident memory or native BLAS scratch storage.

Importing SIMD activates an optional convolution extension only when
`JEFF_CPU_SIMD=1`. It retains tap order and uses no fast-math. Its microbenchmark
was slightly faster, but whole-forward improvement was not demonstrated;
keep it disabled for the reported configuration. A LoopVectorization `@turbo`
convolution trial was slower and is not used in inference.

For experimental tuning, `JEFF_CPU_DELTA_WORKERS` limits workers to the
available Julia threads and value heads (default: available threads), and
`JEFF_CPU_DELTA_CHUNK_SIZE` selects a positive chunk size (default: 64).
Changing chunk size changes floating-point accumulation order and requires
reference validation. `tools/sweep_cpu_delta.jl` screens 1/2/4/8 workers and
16/32/64/128 chunks, using five samples per configuration; its p95 is a
screening statistic, not a reliable tail estimate. Use independent, longer
full-forward repeats before adopting a setting. These controls are not an
automatic performance policy for other CPUs, lengths, batches, or checkpoints.

Two further experiments remain disabled by default. Importing Octavian and
setting `JEFF_CPU_OCTAVIAN_DELTA=1` uses its serial kernel for only two small
worker-owned state products, without changing BLAS threading globally; large
MLP products still use BLAS. The 20-call median was 410.99 ms, not evidence of
a substantial whole-model improvement. The benchmark asserts that the extension
actually loaded, rather than relying on the environment flag alone.

`JEFF_CPU_RECURRENT_DELTA=1` replaces chunk products and triangular solves with
token-wise Float32 state updates fused in column-major SIMD loops. It resets
state per head and token scratch per token, without persistent caches. On the
same input, its 20-call median was 408.83 ms / p95 449.15 ms, with 291,769,968
Julia heap bytes / 5,068 allocations and maximum reference logit error
`9.54e-6`. Allocation counts improved much more than latency. The two paths use
different summation orders: tiny-model reference, mask/length tests, and this
single real input do not establish accuracy on every checkpoint or dataset.
Neither experiment establishes the 2× target.
An independent 50-call recurrent repeat measured 402.39 ms median / 455.79 ms
p95 with the same allocation figures and reference error: about 1.23× the
original Intel baseline, still slower than the required 247.63 ms.

### Portable activation and projection experiment

Importing LoopVectorization enables the optional extension when
`JEFF_CPU_PORTABLE_VECTOR_MATH=1`. It vectorizes owned Float32 SiLU and fused
MLP gating only after checking a conservative finite input domain; exceptional
values, zeros, tiny magnitudes, unsupported types, and gate/up aliasing fall
back without first modifying either input. The ordinary path preserves `up`.
Fast-math can change rounding even inside the accepted domain. This experiment
is disabled by default, and does not prove accuracy for arbitrary models.

`JEFF_CPU_PARALLEL_PROJECTIONS=1` splits output rows across Julia workers,
but only with single-threaded BLAS and sufficiently large products (at least
256 output rows and one million multiply-elements). Set BLAS threads before
inference; the implementation does not change global BLAS settings during a
forward. Aliasing destinations use a temporary result. Inputs and weights
remain read-only, and workers own disjoint output rows until `@sync` completes.

On the same Intel model/input, with recurrent DeltaNet, parallel heads, MLP
and Delta workspace, trim and final-query enabled, 20 warm calls measured:

| Configuration | Median | p95 | Julia heap bytes / allocations |
| --- | ---: | ---: | ---: |
| Julia 8 / OpenBLAS 1, parallel projections, scalar activation | 475.59 ms | 509.22 ms | 292,331,056 / 12,852 |
| Julia 8 / OpenBLAS 1, parallel projections, vector activation | 365.62 ms | 391.09 ms | 292,332,400 / 12,894 |
| Julia 8 / OpenBLAS 8, unsplit projections, vector activation | 513.78 ms | 537.36 ms | 291,775,952 / 5,255 |

The vector/parallel result has maximum saved-reference logit error `1.05e-5`,
model-retained Julia memory 3.01 GB, and process peak RSS 5.35 GB. Peak RSS
includes loading and compilation, not only warmed inference. Raw results are
`artifacts/cpu-tuning/parallel-portable-vector.json`,
`parallel-scalar-repeat.json`, and `openblas8-portable-vector.json`.
The libraries are portable to Linux/macOS, but only this macOS Intel machine
has been measured. Compared with the original 495.25 ms baseline this is
about **1.35×, still not 2×**. Do not replace that baseline with the slower
OpenBLAS control to claim completion.
An independent 20-call vector/parallel repeat measured 382.68 ms median /
462.79 ms p95, with the same heap counts and reference error. Its noisier tail
is a reason not to describe this configuration as universally faster.

Run `tools/verify_cpu_vector_math.jl` in the tools environment for activation
guards, ownership and independent tiny-model checks;
`tools/compare_cpu_vector_math.jl` and `tools/compare_parallel_gemm.jl` are
microbenchmarks, not substitutes for full inference measurements.

### Forward thread policy and full-attention head experiment

Two further opt-ins avoid overhead without changing global BLAS settings:
`JEFF_CPU_PROJECTION_THREAD_SCOPE=1` snapshots the current BLAS thread count
inside one CPU forward using task-local storage, restoring any previous value
even on an exception. It requires parallel projections to be enabled; direct
projection calls outside the scope still check BLAS normally. BLAS configuration
must remain unchanged while any forward is running.

`JEFF_CPU_PARALLEL_FULL_HEADS=1` parallelizes ordinary full-attention heads
only for at least 16 tokens, multiple available workers, and single-threaded
BLAS. Workers read shared Q/K/V/gate/mask and own distinct output head rows
and private scores/probabilities. Q/K/V views avoid head materialization.
The final-query-only path and generic/GPU paths keep their previous computation.
Short sequences and multithreaded BLAS fall back to the original head loop.

With both flags added to the portable configuration above, 20 warm forwards
measured **319.04 ms median / 345.54 ms p95**, with 271,369,904 Julia heap bytes
and 12,671 allocations (`artifacts/cpu-tuning/full-heads-parallel.json`).
This is approximately **1.55× the original baseline, not 2×**. Scope alone
measured 356.96 ms, but its small latency difference needs stronger repeat
evidence before attributing the whole gain to that change.
An independent 30-call repeat measured **318.66 ms median / 333.54 ms p95**
with identical heap figures and maximum reference logit error `1.05e-5`.
Model-retained memory was 3.01 GB and process peak RSS 5.31 GB in that repeat.
The full test suite passed, including mask/length, simultaneous head calls,
input preservation, GC reuse, and task-local policy restoration tests.
JET reported no errors for the inspected real-model forward; its warmed
profile measured 300.78 ms, but is not a substitute for the benchmark trial.

### Delta normalization and blockwise SiLU experiment

`JEFF_CPU_DELTA_NORM_LOOP=1` normalizes owned Q/K arrays with column-major
SIMD reductions and division, avoiding intermediate reduction/broadcast
arrays. It retains Float32, with no explicit fast-math, but the summation
order may change. Reference and exceptional-value checks remain necessary.

`JEFF_CPU_VECTOR_MATH_BLOCKS=1`, together with the portable vector-math
extension, handles SiLU arrays that fail the whole-array guard in blocks of
256 elements. Each block uses the same conservative domain checks: safe
blocks use vector math; other blocks use the unchanged scalar formula,
including NaN/Inf, signed zeros and underflow. It does not relax the accepted
domain or change the MLP gate's ownership/alias behavior. Both flags default
to off.

Adding both flags to the preceding portable configuration measured
302.63 ms median / 324.11 ms p95 over 20 warm forwards, with 271,128,624 Julia
heap bytes / 12,601 allocations and maximum saved-reference logit error
`1.14e-5` (`artifacts/cpu-tuning/delta-norm-block.json`). This is about 1.64×
the original Intel baseline, still above the 247.63 ms target.
`tools/time_cpu_delta_stages.jl` follows real layer activations and validates
the final scores, but its instrumented phase times and domain scans are
diagnostics, not a replacement for full-forward benchmarking.
An independent 30-call repeat measured 304.16 ms median / 328.89 ms p95,
with the same allocation figures and reference error. JET reported no errors
for the inspected forward; the separate warmed profile was 286.47 ms.

## Real 0.8B checkpoint: the README demo

These benchmarks use **Jeff's actual trained 0.8B weights**, not the tiny test
model. The checkpoint is `mstrasser/Jeff-Qwen3.5-0.8B`, pinned to revision
`0f212b3e72acb4dde3f7da61e925d6ab7f819990`. Native Julia loads its safetensors
directly; no ONNX export is used for these measurements.

We measured the same parcel inquiry as `examples/native_inference.jl` and
`examples/metal_inference.jl`: one decision, 256 input positions, with 101
active tokens and 155 leading padding positions. The decision is `delivery`
with probability about 99.66%.

### How to read the numbers

**Latency is the time for one complete model forward after warm-up.** Smaller
is faster. It includes the trained readout and returning scores to CPU. It
excludes package startup, weight loading, compilation, tokenization, and the
small probability-calibration step in `decide`. Running the demo as a new
process therefore takes longer than the latency in this table.

Measured on Apple M4 on 2026-10-01, Julia 1.13.1, Metal 1.11.1, and PyTorch
2.14.0. All rows use Float32 and 20 measured forwards, except the vector-math and parallel-head rows (50). OpenBLAS and PyTorch
CPU use 8 threads; Apple Accelerate uses its framework-managed threading
(10 threads reported on this M4). The LBT/OpenBLAS thread count alone does
not describe Accelerate's actual thread count.
Backends were run separately. Speed ratios compare CPU with CPU and GPU with GPU.

| Backend | Computed tokens | Median per decision | p95 | Speed vs corresponding Python backend |
| --- | ---: | ---: | ---: | ---: |
| Original Python / PyTorch CPU | 256 | 3,016 ms | 3,024 ms | 1× |
| Native Julia CPU, OpenBLAS | 256 | 1,529 ms | 1,593 ms | **1.97×** |
| Native Julia CPU, OpenBLAS + padding trim | 101 | 671 ms | 869 ms | **4.49×** |
| Native Julia CPU, Apple Accelerate | 256 | 533 ms | 573 ms | **5.66×** |
| Native Julia CPU, Apple Accelerate + padding trim | 101 | 257 ms | 274 ms | **11.72×** |
| Native Julia CPU, Accelerate + trim + vector math | 101 | 219 ms | 238 ms | **13.78×** |
| Native Julia CPU, above + MLP workspace + 4 parallel workers | 101 | 190 ms | 198 ms | **15.85×** |
| Native Julia CPU, above + MLP workspace + 8 parallel workers | 101 | 186 ms | 202 ms | **16.24×** |
| Original Python / PyTorch MPS | 256 | 364 ms | 375 ms | 1× |
| Native Julia Metal, default | 256 | 197 ms | 204 ms | **1.85×** |
| Native Julia Metal, workspace + padding trim | 101 | 89 ms | 93 ms | **4.07×** |

For this input, default Metal takes about 46% less time than Python MPS.
The optional configuration takes about 75% less time: workspace buffers are
reused, and the leading padding is removed before computation. Both use the
same logical input and agree with the independent Python reference. The
largest absolute Julia logit error for this demo in these runs was `1.24e-5`.

The 89 ms result requires `JEFF_METAL_WORKSPACE=1` and
`JEFF_METAL_TRIM_PADDING=1`; it is **not the default demo setting**. All other
Metal optimization flags were disabled. This comparison is with the original
Jeff PyTorch implementation, using its reference DeltaNet/convolution fallback
kernels without FLA or causal-conv1d. It does not compare against MLX or every
optimized Python implementation. Different prompts, lengths, batches, and
hardware can produce different ratios.

### Does removing padding change accuracy?

Padding trim removes only the **leading positions whose attention mask is zero**.
It preserves every active token, their order, and interior mask holes. It does
not truncate the prompt. For this checkpoint's causal attention and masked
DeltaNet path, the implementation has been checked against independent PyTorch
scores, including padded inputs and interior mask holes.

In this demo, trimming kept the `delivery` decision and its approximately
99.66% probability. Maximum absolute logit error versus the independent
reference was `7.63e-6` with trimming and `6.68e-6` without it. These differences
are within the numerical verification tolerance; outputs are not bit-identical.
This is evidence of numerical agreement on the tested inputs, not a large-scale
classification-accuracy evaluation or a guarantee for other model architectures.

### First run versus repeated inference

In these runs, native CPU weight loading took 0.77 s and its first forward
took 4.88 s with OpenBLAS. Default Metal loading took 2.25 s and its first forward took
15.29 s. These timings exclude package imports and the checkpoint download;
the first model download is about 1.7 GB. Keep the loaded backend alive when
making repeated decisions to amortize loading and compilation.

### Applying GPU optimization lessons to CPU

CPU profiling found that convolution slice copies and broadcast temporaries
were large allocation sources. The CPU implementation now accumulates
convolution results directly into its output buffer and applies SiLU in place.
As in Metal, the final layer's position-wise MLP computes only the last token;
its attention still reads the full context. JET reported no errors before
the change: type stability alone did not remove the allocation cost.

With 8 BLAS threads and no padding trim, these changes reduced the demo's
median from 1,893 to 1,757 ms and heap allocation bytes from 3.06 to 2.03 GB
per forward. `JEFF_CPU_TRIM_PADDING=1` further reduced the measured median to
674 ms and heap allocation bytes to 0.84 GB. These bytes are cumulative
allocations per call, not resident memory. Trim remains disabled by default
and removes only the leading zero-mask prefix. Maximum absolute logit error
was `1.24e-5` with CPU trimming.

MPS tensor-data and command-queue reuse are specific to Metal. CPU still has
scope for reusable scratch arrays, fewer DeltaNet chunk temporaries, and BLAS
thread tuning. The sampled CPU profile includes triangular solves and BLAS
matrix products, as well as element-wise exponentials and slice copying.
For this trimmed demo, reducing BLAS to one thread increased the median to
1,196 ms, versus 674 ms with eight threads (20 samples each). Fewer threads
are therefore not automatically faster.

```bash
JEFF_CPU_TRIM_PADDING=1 julia --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 20 artifacts/benchmarks/native-cpu-trim.json
# Change benchmark BLAS threads independently; default is 8.
JEFF_BLAS_THREADS=1 JEFF_CPU_TRIM_PADDING=1 julia --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 20 artifacts/benchmarks/native-cpu-trim-blas1.json
# CPU-specific code_warntype, JET, Profile and sampled allocation report:
julia --project=tools tools/profile_native_cpu.jl "$CHECKPOINT" examples/data/parcel_reference.json
```

### Further CPU improvement: DeltaNet chunks and Apple Accelerate

The next CPU change normalizes shared Q/K heads once, uses views for chunks,
performs triangular solves in owned RHS buffers, and uses `mul!` to combine
products and reuse the recurrent state. With OpenBLAS and no trimming, latency
fell further from 1,757 to 1,529 ms, and heap bytes from 2.03 to 1.28 GB. With
trimming, heap bytes fell from 0.84 to 0.50 GB; latency remained about 671 ms
and the p95 was worse in this run. Less allocation does not guarantee lower
latency or better tails.

Following Laya.jl's optional CPU backend, importing
[AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl)
forwards BLAS to macOS Accelerate through libblastrampoline. The model and
its DeltaNet/attention logic remain in Julia; this changes the CPU numerical
library rather than invoking Metal, Python, or ONNX. It is a process-wide BLAS
switch and requires macOS 13.4 or later. The tools environment includes the
measured AppleAccelerate 0.7.0; the package's basic CPU environment does not
require it. Accelerate manages its own threads, separately from OpenBLAS.

With Accelerate, the same demo took 533 ms without trimming. Two 20-sample
trimmed runs measured 254 and 257 ms; the table uses the repeat run.
Maximum demo logit error was `1.15e-5`.
Additional benchmark runs on six independent reference cases covered mixed
English/Japanese, B2 lengths 1/65/512, and B3 interior masks at lengths 65/129.
All passed the numerical check; largest absolute error was `3.61e-5`.
These checks establish agreement on those inputs, not a dataset-wide accuracy
evaluation. The Python comparison is the original fallback implementation and
is not an Accelerate-versus-Accelerate microbenchmark.

```bash
# Fast CPU demo on macOS; set up --project=tools first.
JEFF_CPU_ACCELERATE=1 JEFF_CPU_TRIM_PADDING=1 julia --project=tools examples/native_inference.jl
# Same real model and reference input, 20 samples:
JEFF_CPU_ACCELERATE=1 JEFF_CPU_TRIM_PADDING=1 julia --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 20 artifacts/benchmarks/native-cpu-accelerate.json
```

JSON output records both the loaded BLAS libraries and Accelerate's reported
thread count. The default setting still uses the application's current BLAS;
it does not automatically import AppleAccelerate. Reusing CPU scratch buffers
and reducing remaining normalization/activation costs are further candidates.

### Reproduce the real-checkpoint benchmark

From the repository root, install the tools environment and resolve the same
checkpoint. The bundled `examples/data/parcel_reference.json` contains the
demo's exact token inputs and independently computed original Python logits.
It is measurement data, not a model. No ONNX graph is needed.

```bash
julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
CHECKPOINT=$(julia --project -e 'using JeffClient; print(resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B"; revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990"))')

# CPU: case 1, 20 measured forwards, JSON output.
julia --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 20 artifacts/benchmarks/native-cpu.json
# Apple GPU: default settings (start with no JEFF_METAL_* flags set).
julia --project=tools tools/benchmark_inference.jl "$CHECKPOINT" metal examples/data/parcel_reference.json 1 20 artifacts/benchmarks/native-metal.json
# Apple GPU: optional workspace and leading-padding trim.
JEFF_METAL_WORKSPACE=1 JEFF_METAL_TRIM_PADDING=1 julia --project=tools tools/benchmark_inference.jl "$CHECKPOINT" metal examples/data/parcel_reference.json 1 20 artifacts/benchmarks/native-metal-trim.json
```

For the original Python comparison, install Git and uv and prepare the original
source checkout at the measured commit (skip cloning if it already exists):

```bash
git clone https://github.com/firelex/jeff extern/jeff
git -C extern/jeff checkout f06788292874c21a5b5c41549ac220dd9e15da7f
uv sync --project extern/jeff --frozen --no-default-groups
julia --project=tools tools/benchmark_original.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 20 artifacts/benchmarks/python-cpu.json 1
julia --project=tools tools/benchmark_original.jl "$CHECKPOINT" mps-f32 examples/data/parcel_reference.json 20 artifacts/benchmarks/python-mps.json 1
```

Python is accessed through PythonCall.jl using `extern/jeff/.venv/bin/python`.
Use `mps-f32` for the Float32 comparison; `mps` uses the original BF16 default.
Both tools verify scores before measuring. GPU runs wait for completion.
Julia includes uploading prepared CPU inputs; Python prepares its device inputs
before timing. Both return scores to CPU. Run benchmarks sequentially to avoid
competition for CPU/GPU resources.

JSON output records median, minimum, p95, maximum, loading/first-call timings,
numerical error, and (for Julia) host allocations. Host allocation bytes are
not peak memory usage or GPU buffer bytes. Raw runs are stored in ignored
`artifacts/`; the table above is the versioned summary.

## Earlier measurements and remaining work

On Apple M4, warmed Float32 inference, including CPU score return, measured:

| Prepared input | Julia Metal configuration | Julia median | Original Python MPS median | Speed ratio |
| --- | --- | ---: | ---: | ---: |
| B1/L256, 101 active tokens | Default | 195 ms | 350 ms | 1.8× |
| Same B1 input | Workspace + leading-padding trim | 86 ms | 350 ms | 4.1× |
| B2/L512, 512/256 active tokens | Workspace + trim + shape reuse + fused DeltaNet mask | 582 ms | 1,301 ms | 2.2× |

Model loading, compilation and tokenization are excluded. B1 measurements use
20 samples; the latest B2 Julia measurement uses 50 and Python uses 20. Trimming
reduces actual computation to 101 tokens for B1 and 512/256 for B2; the logical
inputs remain the same. The Python reference uses its fallback MPS kernels,
without the optional FLA or causal-convolution accelerators. These results do
not establish speedups for every shape or Python configuration.

The Julia model, recurrent DeltaNet and full attention are implemented over
Metal. Kernel fusion, cached GPU kernels, command-queue reuse, workspace buffers
and MPS tensor-data reuse remove most of the original host allocation cost.
Independent PyTorch references pass for 15 cases × 3 repeats, including padding
and interior mask holes. The six inspected JET targets report no errors; warmed
allocation profiles record no new MPS tensor-data wrappers with workspaces.
Allocation reduction alone does not guarantee a latency improvement.

Joint GPU batching remains experimental and disabled by default. It speeds up
equal-length B2/L1 from 43.9 to 22.8 ms and reduces allocations from 7,751 to
3,947, but shared padding makes the measured mixed-length L65 and L512 cases
slower than row-wise trimming. The batch prototype has passed independent model
and primitive checks, while further configuration validation remains open.

Remaining work is tracked in:

- [#1: batch applicability, configuration and ownership validation](https://github.com/AtelierArith/JeffClient.jl/issues/1)
- [#2: reduce GPU workspace retention with layer buffer/tensor-data reuse](https://github.com/AtelierArith/JeffClient.jl/issues/2)
- [#3: reduce remaining kernel-submission heap allocations](https://github.com/AtelierArith/JeffClient.jl/issues/3)

Detailed measurement history follows; findings and conditions are recorded in
[measurement notes](https://github.com/AtelierArith/JeffClient.jl/blob/main/memories/MEMORY.md).

## Measurement history

The following snapshots record earlier optimization stages; their settings
and figures should not be combined with the current summary above.

Apple M4, Julia 1.13.1, Metal 1.11.1, PyTorch 2.14.0; pinned Jeff 0.8B,
batch 1, sequence length 256 (101 active tokens), same prepared English prompt,
20 warmed forward passes for current Julia Metal and PyTorch MPS Float32;
the earlier CPU and BF16 runs used 10. Backends were measured separately. Inputs are
prepared before timing. Loading/tokenization are excluded; readout and returning
scores to CPU are included. GPU measurements wait for completion.

| Backend | Weight precision | Samples | Median per forward |
| --- | --- | ---: | ---: |
| Native Julia CPU (8 BLAS threads) | Float32 | 10 | 1.834 s |
| Native Julia Metal (default settings) | Float32 | 20 | 0.195 s |
| Original Jeff / PyTorch CPU (8 threads) | Float32 | 10 | 2.988 s |
| Original Jeff / PyTorch MPS | BF16 (original default) | 10 | 0.333 s |
| Original Jeff / PyTorch MPS | Float32 | 20 | 0.350 s |

For this prepared batch-1 case, Julia Metal takes about **44% less time than
PyTorch MPS at the same Float32 precision** (~1.80× throughput). This comparison
does not establish performance for other lengths or batch sizes.
The Python runs call the original Jeff `forward` with its backbone wrapper,
readout, and option masking. They use the installed reference DeltaNet and
convolution implementations, without Flash Linear Attention or causal-conv1d.

Julia Metal's model loading took 2.04 s and its first forward took 10.26 s
(including compilation, excluding package imports); subsequent timing is above.
BenchmarkTools measured 8,446 Julia heap allocations / 364,512 bytes per
Metal forward; this does not measure GPU buffer bytes. Host profiling includes
MPS submission, Objective-C calls, array allocation, and synchronization.

Following the Laya-based buffer changes and kernel fusion, median Metal latency
fell from 2.492 s to 0.195 s (~12.8× faster). Allocation counts fell from 2,845,912
to 8,446 (~99.7%), and Julia heap bytes from 145,768,352 to 364,512 (~99.7%). Device-only buffers
are reused only on their owning queue; queue roots keep their last references
alive until GPU completion. Shared uploads are not rewritten until a completed
download or explicit synchronization permits recycling. The product graph
overwrites destinations without reading their previous contents.

RoPE/head fusion reduced allocations from 44,097 to 33,202 and heap bytes from
3.46 MB to 1.64 MB. Median latency fell from 253 to 234 ms; the latest trial's
minimum/p95/maximum were 218/312/382 ms. Reusing MPS shapes then reduced
allocations to 28,426 and heap bytes to 1.45 MB; its median/minimum/p95/maximum
were 219/216/291/365 ms. The timing distributions overlap, so the incremental
latency improvement is less certain than the allocation reduction.
Fixed feed/result keys and direct dictionary construction subsequently reduced
allocations to 24,048 and heap bytes to 1.17 MB. The 219 ms median was unchanged;
this step improved host allocation without an observed speed gain.

Mask reuse, last-column final RMS, residual/RMS fusion, and MLP SiLU/up fusion
then reduced allocations to 21,943 / 1.03 MB after removing unused RMS arguments.
Cached decay coefficients and a shared per-row device mask then reduced
allocations to 20,318 / 0.94 MB. Reading packed Q/K directly while normalizing,
and reading packed V in the recurrent kernel, reduced this to 17,012 / 0.76 MB.
RMS/SiLU gate fusion, in-place residual/MLP activation, and fixed-length MPS
pointer storage then reduced allocations to 14,179 / 0.65 MB. That trial's
median/minimum/p95/maximum were 203/198/217/369 ms. These allocation reductions
did not produce a confirmed incremental speed gain; tail latency remains variable.
Weight tensor-data reuse and dedicated embedding gather subsequently reduced
the default path to 13,850 allocations / 638,720 bytes. Its median/minimum/p95/
maximum were 200/198/207/243 ms. The opt-in workspace measurements are described
on the [profiling and tuning page](profiling.md).
Paired Q/K normalization then reduced the default path to 13,213 allocations /
621,344 bytes, with a 200 ms median. This measurement uses commit `1156f4a`;
later beta/decay fusion experiments are not included in these default figures.
After beta/decay fusion and dedicated mask/MLP/residual kernels, the current
default path measured 11,185 allocations / 461,808 bytes, with median/minimum/
p95/maximum 194.9/193.2/201.1/243.4 ms over 20 runs.
Packing RMS launch parameters subsequently reduced the default allocation
count to 10,984 with unchanged 461,808 heap bytes. Median/minimum/p95/maximum
were 194.9/192.7/202.7/249.4 ms; a further latency improvement is not established.
Layer-boundary residual/RMS fusion and readout-column views reduced the current
default path to 10,513 allocations / 457,232 bytes, with median/minimum/p95/
maximum 195.0/192.4/201.0/213.1 ms.
Reusing the MPS wrapper within each open Metal command buffer reduced the
default path to 10,327 allocations / 451,280 bytes. Median/minimum/p95/maximum
were 194.7/192.6/201.9/206.2 ms; latency remained similar.
Removing redundant scalar size arguments from small kernels reduced the
current default path to 10,249 allocations / 448,304 bytes, with median/minimum/
p95/maximum 195.2/192.7/203.0/209.5 ms.
Reusing compiled handles for MLP gates and DeltaNet gates/masks reduced the
current default path to 10,129 allocations / 446,384 bytes, with median/minimum/
p95/maximum 194.9/192.9/201.8/217.4 ms. Handles are invalidated on Julia method
updates; stateful kernel closures use Metal's usual compiler lookup.
Extending handle reuse to common RMS widths, depthwise convolution, softmax,
and attention gates reduced the current default path to 9,130 allocations /
399,360 bytes, with median/minimum/p95/maximum 194.9/192.1/202.5/233.6 ms.
Latency remains similar; these changes reduce host management allocations.
Specialized Q/K, recurrent, and RoPE launches and direct matrix gate outputs
reduced the latest default path to 8,446 allocations / 364,512 bytes, with
median/minimum/p95/maximum 195.2/192.5/207.9/214.8 ms.

The latest free private cache snapshots were 1.65/6.05/4.77 GB after
trial/full GC/explicit trim. The cache limit is one quarter of recommended
working set, checked on allocation pressure and completed score downloads.
Delayed GC returns can exceed it until the next trim. These snapshots are not
peak or total resident GPU memory. Shared uploads are limited to 64 MB per queue.

A second matched Float32 comparison used batch 2 / length 512 with 512 and 256
active tokens per row: current Julia Metal with `JEFF_METAL_WORKSPACE=1` measured
**776 ms**, original Python MPS **1,301 ms** (20 samples each), about 40% less time.
Julia measured 8,328 host allocations / 455,392 bytes per forward.
The workspace holds 447 arrays / 1,766,096,896 device-buffer bytes; these exclude
weights and native MPS resources. Julia currently processes batch rows sequentially.
This result supports that specific longer input; other shapes and optimized
Python kernels still need separate measurements.

```bash
julia --project=tools tools/benchmark_inference.jl CHECKPOINT_DIRECTORY metal artifacts/jeff-0.8b-onnx/reference.json 1 20 artifacts/metal-validation/benchmark-metal.json
julia --project=tools tools/benchmark_inference.jl CHECKPOINT_DIRECTORY cpu artifacts/jeff-0.8b-onnx/reference.json 1 10 artifacts/metal-validation/benchmark-cpu.json
# Original Python resources are called through PythonCall.jl:
julia --project=tools tools/benchmark_original.jl CHECKPOINT_DIRECTORY mps-f32 artifacts/jeff-0.8b-onnx/reference.json 20 artifacts/metal-validation/benchmark-python-mps-f32.json
```

`benchmark_original.jl` also accepts `cpu` and `mps` (original BF16 default).
It accepts an optional case index after the output path, and both benchmark tools
can use the expanded reference document. `tools/benchmark_stages.jl` measures
synchronized individual stages; those costs are not additive to a full forward.
Set `JEFF_PROFILE=1` when running `benchmark_inference.jl` for a warmed CPU
sampling profile. Raw JSON measurements are kept under gitignored `artifacts/`.

### CPU buffer reuse and optional vector math

CPU DeltaNet now reuses chunk scratch matrices across heads and chunks within
each layer call. On the same real 0.8B parcel input, allocation fell from
504,848,576 bytes / 30,103 allocations to 385,530,176 bytes / 13,903 allocations.
Latency remained about 256 ms. These buffers belong to the individual forward.

On macOS, `JEFF_CPU_VECTOR_MATH=1` enables the optional AppleAccelerate extension
for vector exponentials in SiLU and MLP gating. Disposable projection arrays
provide scratch storage. With Accelerate and padding trimming, 50 warmed
forwards measured median **218.832 ms**, p95 **237.897 ms**, and
385,531,520 heap bytes / 13,945 allocations. This is about 14.6% less time than
the preceding 256 ms configuration, and 13.78× faster than the measured
original Python CPU fallback (3,016 ms). The comparison includes different
math kernels and less padding work; it is not a language-only comparison.

```sh
JEFF_CPU_VECTOR_MATH=1 JEFF_CPU_ACCELERATE=1 JEFF_CPU_TRIM_PADDING=1 julia --project=tools examples/native_inference.jl
JEFF_CPU_VECTOR_MATH=1 JEFF_CPU_ACCELERATE=1 JEFF_CPU_TRIM_PADDING=1 julia --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 50 artifacts/benchmarks/native-cpu-vector.json
```

Both chunk reuse and vector math passed independent PyTorch logit guards on
15 prepared cases, including batch inputs, chunk boundaries and interior mask
holes. Maximum absolute errors were 3.60e-5 and 3.40e-5 respectively. These
checks do not establish dataset accuracy or arbitrary-input numerical safety.
Vector math remains disabled by default pending broader validation, including
extreme activation values and destructive scratch ownership.

The separate experimental `JEFF_CPU_INPLACE_DELTA_RMS=1` reduced allocations
to 369,828,032 bytes / 9,931 allocations, with median 218.046 ms over 50
forwards. Timing overlaps the vector-only run, so no additional speedup is
claimed. It remains disabled by default. Setting
`JEFF_CPU_ACCELERATE_THREADS=1` in the benchmark gave 219.573 ms; single-thread
Accelerate did not improve this input. Values above one select framework
managed threading, rather than an exact thread count.

Remaining CPU work: [workspace reuse (#4)](https://github.com/AtelierArith/JeffClient.jl/issues/4),
[MLP layout and parallelism (#5)](https://github.com/AtelierArith/JeffClient.jl/issues/5),
and [vector/RMS validation (#6)](https://github.com/AtelierArith/JeffClient.jl/issues/6).
RMS also passed the 15 prepared-case guards (maximum absolute logit error
3.40e-5); broader ownership and numerical validation remains.

### CPU MLP workspace and parallel DeltaNet heads

`JEFF_CPU_MLP_WORKSPACE=1` reuses gate/up projection buffers across layers
within one forward. The final MLP has separate one-token buffers. On the
parcel input, this reduced heap allocation from 385.5 MB to 321.4 MB; median
latency remained about 222 ms with the new overflow guard.

`JEFF_CPU_PARALLEL_HEADS=1` distributes independent DeltaNet heads across
Julia workers. Each worker owns its state and scratch matrices; output slices
do not overlap. The layer waits for all workers before its output projection.
Both options are disabled by default and allocate their workspace locally to
the forward, rather than caching mutable arrays on the backend.

With Accelerate, vector math, trimming and MLP workspace enabled, 50 forwards
on the same Apple M4 / Float32 / real 0.8B input measured:

| Head execution | Julia workers | Median | p95 | Heap bytes | Allocations |
|---|---:|---:|---:|---:|---:|
| Serial | 4 | 225.306 ms | 240.565 ms | 321,368,912 | 13,833 |
| Parallel | 4 | 190.265 ms | 198.349 ms | 348,846,704 | 17,775 |
| Parallel | 8 | 185.748 ms | 201.600 ms | 385,451,504 | 22,671 |

Four workers reduced median time by 15.6% against the same-worker serial run.
Eight workers were slightly faster but allocated more, and had a higher p95.
Accelerate single-thread mode with eight Julia workers measured 188.755 ms,
so it was not adopted as the fastest setting.

Parallel execution passed 15 independent prepared-reference cases and
shape/mask revisits, with two calls per case including a call after full GC.
Inputs and previously returned scores stayed unchanged; maximum absolute
logit error was 3.40e-5. JET reported no errors for core and extension methods.
The sampled parallel profile measured a warm forward of 186.076 ms. These
checks establish the tested cases, not dataset accuracy or all possible inputs.

```sh
JEFF_CPU_PARALLEL_HEADS=1 JEFF_CPU_MLP_WORKSPACE=1 JEFF_CPU_VECTOR_MATH=1 \
JEFF_CPU_ACCELERATE=1 JEFF_CPU_TRIM_PADDING=1 \
julia --threads=8 --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 50 artifacts/benchmarks/native-cpu-parallel.json
```

Diagnostic tools:

- `tools/inspect_native_cpu_types.jl CHECKPOINT REFERENCE_JSON [all|hidden|workspace|layer|mlp|gate] [typed|source|descend]` inspects CPU types; `descend` requires a terminal.
- `tools/benchmark_cpu_projections.jl CHECKPOINT REFERENCE_JSON [SAMPLES] [OUTPUT_JSON]` measures representative projections and sweeps through all MLP weights, excluding activation and attention logic.
- `tools/validate_native_cpu.jl CHECKPOINT REFERENCE_JSON` checks the extended 15-case reference, repeat calls, GC and input/score ownership.

Cthulhu descent through MLP and `mul!` found concrete array and return types.
The small `Nothing`/workspace union is narrowed in the consuming branch.
No type-instability fix is claimed. Weight materialization, transposed MLP
layout and combined gate/up packing were measured and rejected because they
did not demonstrate a speed improvement. Parallel scratch reuse across layers
remains a candidate for reducing the increased allocation.

### Portable CPU Delta projection workspace (Intel)

`JEFF_CPU_DELTA_PROJECTION_WORKSPACE=1` reuses DeltaNet projection and
preparation buffers across layers within a forward. It is disabled by default.
The buffers are not shared between forwards; convolution scratch is cleared
before accumulation and projection outputs are overwritten.

On Intel i9-9900K, Julia 1.13.1, Float32, Julia eight workers and OpenBLAS one
thread, the real 0.8B parcel input (batch one, padded 256, active 101) measured
290.285 ms over 20 warmed forwards and 296.796 ms over an independent 30-forward
repeat. This used the portable vector/block, Delta normalization/recurrent,
parallel projection/head/full-head, projection-scope, MLP/Delta workspace,
padding-trim and final-query options. No MKL or Accelerate was used.
The preceding configuration measured 304.157 ms. Heap allocation decreased
from 271,128,624 to 113,553,680 bytes per forward, with 11,895 allocations.
The projection workspace itself contains 9,114,240 bytes of array payload on
this input; this is distinct from cumulative heap allocation or process RSS.

Maximum absolute error against the saved independent PyTorch reference was
1.1444e-5. Shape/mask reuse, scratch overwrite, GC, concurrent-forward and
input/returned-score ownership checks passed, as did the full test suite and
JET checks. These results do not establish zero allocation or dataset accuracy.
Against the fixed 495.254 ms pre-optimization Intel baseline, the repeat is
about 1.67 times faster; the two-times target remains unmet.
