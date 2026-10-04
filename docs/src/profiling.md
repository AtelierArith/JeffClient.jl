# Profiling and measurements

The tables below report measured inference latency. CPU configuration and
benchmark entry points are described in [Performance](performance.md).

## Apple M4: matched comparisons (2026-10-01)

All previous Apple M4 benchmark tables, speed ratios and benchmark JSON in
this repository's documentation have been withdrawn. The measurements below
are new; they are independent of the retained Linux/Intel history.

### Common model and timing conditions

Apple M4 (10 CPU cores), 24 GiB RAM, arm64 macOS (Darwin 27.0.0), AC power;
Julia 1.13.1, Python 3.13.15, PyTorch 2.14.0, Transformers 5.17.0, Metal 1.11.1,
AppleAccelerate 0.7.0, MLX 0.32.3 and mlx-lm 0.31.3.
The inference source is `bcc8db9`, with the benchmark-only adapters linked below.
The original Jeff source is `f067882`. The local `models/jeff-0.8b` checkpoint
is identified by complete model/readout/config SHA-256 hashes in the
[new measurement JSON](assets/benchmarks/m4-2026-10-01-matched.json).

Every timed forward uses parcel case 1, batch 1, **Float32 inference and all
256 positions**, including 155 leading padding positions and 101 active tokens.
Weights are widened from the same checkpoint before inference. Readout computes
255 scores; the single-thread and GPU adapters request all options from PyTorch.
Padding trim, last-query-only attention and last-token-only MLP are disabled.
The final residual/RMS also processes all columns before selecting the readout
column. No persistent recurrent/KV cache or cross-request prompt cache is used.

Imports, model loading, tokenization, calibration, input preparation and the
first-call compilation are excluded. Two forwards warm the implementation;
each process then times 30 complete forwards, one forward per evaluation.
Prepared **CPU** inputs, CPU score return and GPU completion are included.
All benchmarks run in separate, sequential processes. CPU input upload is
included for every GPU implementation. Python/MLX use `perf_counter`; Julia
uses BenchmarkTools (`evals=1`). Both collect garbage before the trial, without
forced collection between samples; ordinary runtime GC remains enabled.

The host is not isolated, and CPU affinity is not pinned. Repeated runs report
variation rather than a general confidence interval. Kernels, head batching,
workspace reuse and allocation strategies differ: this compares implementations
of the same model computation, not language overhead. Julia heap allocation
is not comparable to a PyTorch/MLX memory metric and is kept only in the JSON.

### CPU: one thread on both sides

The strict thread comparison sets **Julia workers = BLAS = Accelerate = 1**
and **PyTorch intra-op = inter-op = 1**. Recorded runtime thread counts confirm
these settings. Julia uses Accelerate and the chunk64 reference profile.
PyTorch uses Transformers' CPU reference kernels; FLA and causal-conv1d are
not installed. Each implementation is measured twice in fresh processes.

| Implementation | Median | p95 | Minimum–maximum |
|---|---:|---:|---:|
| Python / PyTorch, run 1 | 545.49 ms | 557.46 ms | 536.69–558.74 ms |
| Python / PyTorch, run 2 | 528.40 ms | 540.35 ms | 514.46–542.73 ms |
| Julia / Accelerate, run 1 | 462.22 ms | 504.21 ms | 461.12–505.32 ms |
| Julia / Accelerate, run 2 | 461.75 ms | 496.45 ms | 460.75–504.22 ms |

Julia's median is about **1.14–1.18× faster** in these two pairs. This is specific
to this checkpoint, input and host. Both implementations pass the saved PyTorch
Float32 reference guard (`atol=rtol=2e-4`) over all 255 logits; the Python
maximum absolute error is zero and Julia's is `6.67572e-6`.
These diagnostic runs do not measure the production padding/last-token shortcuts.

For standard reproduction on Apple Silicon macOS, use the sequential driver:

```sh
./tools/mac-M-series.sh
# Include MLX (the automatic-thread CPU comparison runs by default):
./tools/mac-M-series.sh --mlx
# Prepare the Python environment first if needed (requires uv):
./tools/mac-M-series.sh --setup-python --mlx
```

The default includes both one-thread and automatic-thread CPU comparisons,
with 30 samples × 2 fresh processes per implementation. Use `--help`
for checkpoint/reference/output and sample/repeat options. It saves `summary.md`,
`summary.json`, individual JSON/logs, runtime versions and model/reference/tool
hashes under a timestamped ignored artifacts directory. It verifies actual CPU
thread counts and untrimmed computed lengths before accepting the summary.
The separate one-off commands below show the underlying measurement adapters.

From the repository root:

```sh
julia --threads=1 --startup-file=no --project=tools tools/benchmark_cpu_single_python.jl models/jeff-0.8b cpu examples/data/parcel_reference.json 30 artifacts/benchmarks/m4/python-single.json
julia --threads=1 --startup-file=no --project=tools tools/benchmark_cpu_single_julia.jl models/jeff-0.8b cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/m4/julia-single.json
```

### Additional CPU trial: PyTorch 8 versus Accelerate automatic 10

This is a **different thread budget**, and is not the strict comparison above.
Python requests 8 intra-op threads. Julia starts with 8 workers and requests
BLAS 8, but AppleAccelerate reports **10**, under framework-managed threading.
AppleAccelerate 0.7.0 selects single-thread mode for 1 and automatic mode for
other values; a BLAS/LBT count of 8 does not mean Accelerate is fixed at 8.
These trials still process all 256 tokens and disable the final-layer shortcuts.
The original Python benchmark applies its usual valid-option mask; its error
check covers the valid options rather than all 255 output columns.

| Implementation | Median (run 1 / repeat) | p95 (run 1 / repeat) |
|---|---:|---:|
| Python / PyTorch 8 (one trial) | 3033.09 ms | 3046.57 ms |
| Julia 8 workers / Accelerate automatic 10 | 364.26 ms / 362.11 ms | 381.99 ms / 382.07 ms |

The observed roughly 8.3× ratio includes different thread budgets, backend
kernels and scheduling. It is **not an equal-thread speedup** and is not evidence
that Julia's language runtime is 8.3× faster. PyTorch 8 was slower than its
single-thread trials on this host/input; increasing thread count is not
necessarily beneficial. Automatic Accelerate is not an exact 10-thread pin.

```sh
julia --threads=8 --startup-file=no --project=tools tools/benchmark_original.jl models/jeff-0.8b cpu examples/data/parcel_reference.json 30 artifacts/benchmarks/m4/python-eight.json
julia --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl models/jeff-0.8b cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/m4/julia-auto.json --python-reference
```

### GPU: Metal.jl, PyTorch MPS and MLX

These rows use the same Apple M4 GPU, full 256 positions, Float32 and explicit
completion/CPU score return, with CPU input upload included. They are separate
from the CPU thread comparison. Host execution/threading is framework-dependent.
Metal's workspace, packing, batching, shape-workspace, fused-mask and trim opt-ins
are all off. A benchmark-local method adapter computes the final residual/RMS
for every column; production source is unchanged.

| Implementation | Median (run 1 / repeat) | p95 (run 1 / repeat) |
|---|---:|---:|
| Python / PyTorch MPS Float32 | 343.29 ms / 344.18 ms | 351.32 ms / 350.07 ms |
| Julia / Metal.jl, full-sequence adapter | 197.25 ms / 196.87 ms | 208.95 ms / 204.75 ms |
| MLX GPU, reference-normalization adapter | 136.83 ms / 137.03 ms | 138.61 ms / 138.38 ms |

Metal.jl is about **1.74–1.75× faster** than PyTorch MPS in these pairs.
Maximum absolute errors over 255 logits are `1.00136e-5` for PyTorch MPS,
`6.67572e-6` for Metal.jl and `1.52588e-5` for the adapted MLX GPU; all pass the
same reference tolerance. Metal.jl also passes **15 cases × 2 passes**, including
English/Japanese batches, padded/unpadded sequence boundaries and interior mask
holes, with GC between cases; maximum error is `3.361702e-5` and scalar GPU
indexing is disabled. No large Metal/PyTorch numerical discrepancy was found.
MLX is verified only for the benchmark case, not this broader 15-case suite.

**The MLX row is adapted, not the unmodified upstream Jeff MLX backend.** The
checkpoint weights are cast to Float32 *before* MLX sanitization adds 1 to
normalization weights. The adapter passes padding/causal masks into the original
layers and uses no persistent cache. Transformers adds DeltaNet normalization
`epsilon=1e-6` to the **sum** of squares; mlx-lm 0.31.3 adds it to the **mean**.
The process-local adapter divides MLX's epsilon by head width, making these
formulas equivalent while retaining MLX's original operations/Metal kernels.
The installed package files are unchanged. With the upstream normalization,
the maximum error is about `0.0133` and the guard fails; that run supplies no
accepted latency or speedup. This distinction is included in the raw JSON.

The APIs for [device selection and synchronization](https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html)
are used explicitly; constructing a lazy MLX graph alone is not timed as inference.

```sh
julia --threads=8 --startup-file=no --project=tools tools/benchmark_metal_reference.jl models/jeff-0.8b metal examples/data/parcel_reference.json 1 30 artifacts/benchmarks/m4/metal.json
julia --threads=8 --startup-file=no --project=tools tools/benchmark_gpu_python.jl models/jeff-0.8b mps-f32 examples/data/parcel_reference.json 30 artifacts/benchmarks/m4/pytorch-mps.json
uv pip install --python extern/jeff/.venv/bin/python 'mlx-lm==0.31.3' 'mlx==0.32.3'
julia --threads=1 --startup-file=no --project=tools tools/benchmark_mlx.jl models/jeff-0.8b gpu examples/data/parcel_reference.json artifacts/benchmarks/m4/mlx-gpu.json 30 reference-norm
# Without the final reference-norm argument, the stock-normalization guard fails.
```

### MLX CPU: additional result

The same Float32/full-sequence/reference-normalization adapter also runs on
`mx.cpu`, with CPU inputs and CPU score return included. This is **framework-managed
CPU execution**, not a verified single-thread comparison. One 30-forward trial
measured **2321.83 ms median / 2346.93 ms p95** and passed the reference guard
(maximum absolute error `7.15256e-6`). No equal-thread MLX CPU speed ratio is claimed.

```sh
julia --threads=1 --startup-file=no --project=tools tools/benchmark_mlx.jl models/jeff-0.8b cpu examples/data/parcel_reference.json artifacts/benchmarks/m4/mlx-cpu.json 30 reference-norm
```

The linked JSON contains per-process summaries, available per-sample times,
thread/policy values, source/model/reference hashes, runtime versions and the
Metal verification log. Local setup and full stdout logs are in the ignored
`artifacts/benchmarks/fair-cpu-m4/` directory. The Julia adapters require the
benchmarked source layout and change methods/settings only in isolated processes.

## Linux CPU results (2026-10-01)

### Current matched comparison and allocation investigation

The current eight-physical-core comparison and reproduction commands are in
[Performance](performance.md). Three fresh Julia/MKL runs have medians
767.658/771.419/795.975 ms, versus paired Python/PyTorch
812.884/809.849/805.778 ms. All 255 saved reference logits pass with maximum
error `1.04904e-5`. The third Julia p95 remains higher than Python's.

Before optimization, Julia's diagnostic phase profile concentrated time in
Delta attention and scalar activation fallbacks. A separate PyTorch profile
spent 553.0 of 807.3 ms (68.5%) in GEMM and 59.3 ms in batched GEMM. These
are single-forward diagnostic measurements, not matched benchmark medians.
The measured Python advantage was largely in compiled kernels and array
layout; Python interpreter overhead did not dominate this profile.

Julia improvements include guarded SIMD with scalar repair of exceptional
lanes, contiguous `(width, sequence, head)` Delta storage, in-place softmax,
and forward-owned RMS/residual/full-attention buffers. RoPE tables are reused
within a forward; parallel heads have separate score/value scratch. Reusing
these buffers retains caller input, model weights and returned score ownership.

| Development stage | Estimated heap bytes / forward | Median / p95 (ms) |
|---|---:|---:|
| SIMD / packed Delta / in-place softmax | 324,399,376 | 790.924 / 1006.678 |
| Add RMS/residual reuse | 249,627,968 | 790.073 / 971.406 |
| Add full-attention workspace | 97,363,280 | 794.927 / 925.063 |
| Also normalize owned Delta output in place | 58,739,024 | 766.207 / 804.539 |

These are sequential development trials with MKL1/Julia8, full256/chunk64,
10 warm-ups and 30 samples. They collected GC **after** warm-up; the current
matched driver collects it **before** warm-up on both sides. Do not combine
their ratios with the current Python table. RMS/full-attention reuse chiefly
reduced allocation; the last step also improved latency in this experiment.
The [current evidence JSON](assets/benchmarks/linux-cpu-2026-10-01-matched-mkl.json)
retains the development timing/GC series separately from the final comparison.

Before buffer reuse, slow samples of 1006.7/1110.3 ms included
134.8/141.5 ms of reported GC. After reuse the final three runs report median
GC zero and maximum 12.6–16.9 ms, while their slowest samples report GC zero.
Allocation/GC explains part of the older tail; it does not explain all latency
variation. Heap allocation is cumulative allocation traffic, separate from
retained model memory and process RSS.

The updated native path passes the full test suite, poisoned scratch and
same-length mask changes, RoPE width/length checks, independent fixtures, GC
and returned-score ownership checks. `@code_warntype` returns
`Matrix{Float32}` and JET reports no errors; heap bytes are measured separately
because stable types do not eliminate arrays. These checks cover CPU changes;
Apple Silicon and CUDA results below retain their original measurement scope.

### Earlier baseline: model, input and automatic CPU policy

Source `fddc576013bf1f6971bc7ffdcd901c7d3c44ccab`, with a clean tracked worktree
at measurement time, was measured on **Intel Xeon E5-2699 v3 / x86_64 Linux /
Julia 1.13.1**. Both runs use the real `mstrasser/Jeff-Qwen3.5-0.8B` checkpoint,
pinned to `0f212b3e72acb4dde3f7da61e925d6ab7f819990`, Float32, and parcel
reference case 1: batch 1, padded length 256, 101 active/computed tokens.

The automatic portable policy uses **8 Julia workers and OpenBLAS 1 thread**.
LoopVectorization 0.12.174 is active for portable SiLU/gate vector math;
Octavian and the explicit SIMD extension are disabled. Each run is a separate
process, executed sequentially, with 30 warmed complete `logits` calls and
one forward per evaluation. Readout and CPU score return are included;
download, imports, model loading, tokenization and first-call compilation are
excluded from warm latency.

| Run | Median | p95 | Minimum | Maximum | Julia heap per forward | Allocations |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 448.36 ms | 641.09 ms | 424.77 ms | 641.48 ms | 91.24 MiB | 12,842 |
| 2, independent repeat | 641.81 ms | 799.07 ms | 438.43 ms | 803.77 ms | 91.24 MiB | 12,842 |

There is substantial variation between runs; the faster median is not a stable
latency estimate. The host was not isolated from other workloads. CPU affinity
allowed CPUs 0–35, CPU 0 used the `schedutil` governor, and the process/ancestor
cgroups had no CPU quota or recorded throttling. These observations do not
establish the cause of the variation. Keep both results and compare them only
with matched hardware and conditions.

Both runs passed the saved independent PyTorch reference guard, with maximum
absolute logit error **1.2397766e-5**. Model loading took 3.22/3.18 seconds and
the first forward, including compilation, took 21.39/21.15 seconds. Retained
model memory was 2.80 GiB; process peak RSS was 4.06/4.13 GiB and includes
startup/loading/compilation. Per-forward cumulative Julia heap allocation
(95,672,592 bytes) is a separate metric from retained memory and peak RSS.

The [benchmark JSON](assets/benchmarks/cpu-2026-10-01-fddc576-linux-xeon.json)
preserves both runs, CPU policy values, dependency versions and source/reference
identifiers. Local logs, the resolved Manifest and hardware/cgroup snapshots
are in the ignored `artifacts/benchmarks/cpu-2026-10-01-fddc576/` directory.

```sh
julia --startup-file=no --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
CHECKPOINT=$(julia --threads=8 --startup-file=no --project=tools -e 'using JeffClient; print(resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B"; revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990"))')
julia --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/cpu-2026-10-01-fddc576/native-cpu-8threads.json
julia --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/cpu-2026-10-01-fddc576/native-cpu-8threads-repeat.json
```

### Octavian chunk-path comparison

On the same source, model, input, Float32 precision and Julia8/OpenBLAS1
configuration, the requested diagnostic scope sets `octavian_delta=true` and
`recurrent_delta=false`. Octavian 0.3.29 handles the two worker-local DeltaNet
chunk state products via `matmul_serial!`: the real model has 128×128 state
matrices and chunk width at most 64, satisfying the extension's size guards.
MLP projections and other matrix products continue to use OpenBLAS.

A separate, sequential control run keeps chunk mode and changes only
`octavian_delta` to false. Each run measures 30 warmed complete forwards with
the same timing exclusions as above.

| Delta implementation | Median | p95 | Minimum | Maximum | Julia heap per forward | Allocations |
|---|---:|---:|---:|---:|---:|---:|
| Chunk / Octavian | 753.98 ms | 881.31 ms | 684.41 ms | 886.27 ms | 109.44 MiB | 22,536 |
| Chunk / OpenBLAS control | 752.46 ms | 881.50 ms | 724.98 ms | 882.36 ms | 109.39 MiB | 21,384 |

Octavian's median was 0.2% higher in this pair; this measurement does not show
a speed improvement. The earlier recurrent-mode runs had substantial timing
variation and are not a matched comparison isolating Octavian. Both chunk runs
passed the independent reference guard: maximum absolute logit error was
1.1444092e-5 with Octavian and 1.335144e-5 with OpenBLAS. Diagnostic overrides
were restored after each run; the production defaults are unchanged.

The [comparison JSON](assets/benchmarks/cpu-2026-10-01-fddc576-linux-xeon-octavian.json)
preserves both results and the diagnostic settings. Individual JSON/logs and
the existing `tools/verify_cpu_octavian.jl` verification log are in
`artifacts/benchmarks/cpu-2026-10-01-fddc576/`.

Using the pinned `CHECKPOINT` resolved above:

```sh
julia --threads=8 --startup-file=no --project=tools -e 'using JeffClient, Octavian; JeffClient.with_cpu_settings(:octavian_delta => true, :recurrent_delta => false) do; include("tools/benchmark_inference.jl"); end' "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/cpu-2026-10-01-fddc576/octavian-delta-chunk.json
julia --threads=8 --startup-file=no --project=tools -e 'using JeffClient; JeffClient.with_cpu_settings(:octavian_delta => false, :recurrent_delta => false) do; include("tools/benchmark_inference.jl"); end' "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/cpu-2026-10-01-fddc576/openblas-delta-chunk.json
```

### Python comparison and thread counts

All rows use the same model and Float32 scores, with 30 warmed forwards in
sequential processes. PyTorch uses 8 intra-operation threads; Julia uses the
listed worker/BLAS counts. CPU affinity is unrestricted (CPUs 0–35), and the
host has other workloads. These are individual trials, not isolated-host
confidence intervals.

| Implementation | Julia / BLAS threads | Computed tokens | Median | p95 |
|---|---:|---:|---:|---:|
| Original Python / PyTorch 8 | — | 256 | 808.50 ms | 812.72 ms |
| Julia automatic, fresh comparison run | 8 / 1 | 101 | 652.82 ms | 813.77 ms |
| Julia automatic | 1 / 8 | 101 | 945.12 ms | 975.63 ms |
| Julia Octavian chunk | 8 / 1 | 101 | 753.98 ms | 881.31 ms |
| Julia Octavian chunk | 1 / 8 | 101 | 974.03 ms | 1027.08 ms |
| Original Python, leading padding removed | — | 101 | 432.20 ms | 434.21 ms |

The 808.50/652.82 ratio (1.24×) includes Julia skipping 155 padding tokens
and reducing work in the final layer. It is **not an equal-work comparison**.
Python with the same leading padding removed measured 432.20 ms; this also
shows why the padded Python result cannot establish a Julia speed advantage.
Julia 1 / OpenBLAS 8 was slower than Julia 8 / OpenBLAS 1 in these trials.

The installed FLA package selected a GPU-only Triton implementation on CPU,
so the initial Python run failed. The successful runs select the original
Transformers `torch_chunk_gated_delta_rule` and
`torch_recurrent_gated_delta_rule` functions with `inspect.unwrap` in the
benchmark process, and disable hub kernels with `USE_HUB_KERNELS=NO`.
The Python model runs through PythonCall.jl; no Python source files were changed.
Environment: Python 3.12.3, PyTorch 2.14.0+cu130, Transformers 5.17.0,
FLA 0.5.2; original Jeff commit `f06788292874c21a5b5c41549ac220dd9e15da7f`.

[Thread comparison JSON](assets/benchmarks/cpu-2026-10-01-fddc576-linux-xeon-python-threads.json)
records the padded Python and Julia trials. The trimmed Python result and the
full-sequence comparison below are in the
[reference comparison JSON](assets/benchmarks/cpu-2026-10-01-linux-xeon-python-reference.json).

### Matching full-sequence work with `--python-reference`

The Julia benchmark now provides a CPU-only `--python-reference` profile.
It disables padding trim, final-query-only attention and final-token-only
MLP/RMS, and uses chunk64 DeltaNet with OpenBLAS (Octavian off). Every layer
processes the same 256 positions as the Python reference. Normal inference
keeps its existing optimized defaults; the settings apply only within this
benchmark call and are restored afterward.

| Full-sequence implementation | Median | p95 | Maximum absolute logit error |
|---|---:|---:|---:|
| Python / PyTorch 8 | 808.50 ms | 812.72 ms | 1.0490e-5 |
| Julia 1 / OpenBLAS 8 / reference profile | 1797.10 ms | 1969.85 ms | 1.2398e-5 |

**Python was 2.22× faster in this comparison.** Julia allocated 378,678,568
heap bytes (361.14 MiB) and 20,951 allocations per forward. Both runs passed
the saved independent PyTorch reference guard. Unlike the earlier default
comparison, sequence length and final-layer work match. Kernel implementation,
head batching, workspace reuse and timing harness still differ; this result
does not isolate language overhead. The host was not isolated or pinned.

The Julia measurement uses the diagnostic changes based on `fddc576`, rather
than that clean commit alone. The linked reference JSON records source-file
hashes and both raw summaries. The full test suite passes, including 52 checks
of the full-sequence path against independent references, workspace/no-workspace,
GC reuse, input ownership and restoration of defaults.

```sh
julia --threads=1 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/python-reference.json --python-reference
```

To reproduce the Python CPU run in the installed FLA environment, select its
original CPU reference functions through PythonCall before including the tool:

```sh
USE_HUB_KERNELS=NO julia --threads=8 --startup-file=no --project=tools -e '
include("tools/python_env.jl")
using PythonCall
pyexec(raw"""
import inspect
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
    reference = inspect.unwrap(getattr(qwen, name))
    assert reference.__module__ == qwen.__name__
    setattr(qwen, name, reference)
""", pydict())
include("tools/benchmark_original.jl")
' "$CHECKPOINT" cpu examples/data/parcel_reference.json 30 artifacts/benchmarks/python-cpu.json 1
```


### CUDA / ONNX Runtime verification

Native Julia CUDA inference is now available through the optional CUDA extension;
see the native measurements below. The following paragraph records the earlier
environment failure before the driver update.
On 2026-10-01, CUDA.jl 6.4.1 reported `CUDA.functional() == false`, and a CuArray
round trip fails with CUDA error 804. The loaded NVIDIA driver (580.173.02)
and userspace libraries (580.178.04) differ. ONNX Runtime's tiny CPU fixture
passes, but CUDA session construction fails; real-model GPU inference was not
measured. The selected CUDA runtime 13.4 is also outside ONNXRunTime.jl 1.4.0's
CUDA 12.x range. This is a blocked environment check, not a GPU speed result.
See [CUDA setup and verification](inference.md#CUDA-through-ONNX-Runtime) and
[verification JSON](assets/benchmarks/cuda-2026-10-01-fddc576-linux.json).

On 2026-10-02, NVIDIA driver 580.178.04 initialized successfully on the RTX 3060
host. CUDA.jl 6.4.1 passed a synchronized CuArray round trip. With runtime 13.4,
ONNXRunTime.jl 1.4.0 still rejected session construction due to its CUDA 12.x
requirement. Selecting runtime 12.8 in the isolated validation environment and
restarting Julia resolved this. Both the repository's identity fixture and
ONNXRunTime's MatMul fixture passed through `ONNXBackend(...;
execution_provider=:cuda)`. The non-square Float32 matrix product (2×3 times
3×4) matched Julia, and `decide` returned the expected choices for both rows.
Verbose ONNX Runtime logging confirmed all nodes (one MatMul node) were placed
on `CUDAExecutionProvider`. Local logs are retained under ignored
`artifacts/cuda-validation/onnx-2026-10-02.log` and
`artifacts/cuda-validation/onnx-provider-2026-10-02.log`.

The full `mstrasser/Jeff-Qwen3.5-0.8B` checkpoint at revision
`0f212b3e72acb4dde3f7da61e925d6ab7f819990` was subsequently exported using
`tools/_onnx_export.py` through PythonCall. Fixed-shape Float32 export used
batch 1 and sequence length 256. The parcel prompt has 101 active tokens and
155 left-padding tokens. All 255 logits passed comparison against independent
PyTorch outputs for three prompts (`atol=2e-3`, `rtol=2e-3`); maximum absolute
CUDA errors were 0.003813, 0.002560 and 0.001807. The CPU export validation's
maximum absolute error was 6.68e-6. ORT assigned 19,921 nodes to CUDA and 24
to CPU and inserted 52 copy nodes; this is mixed provider execution.

With verbose logging disabled, five warmups followed by 30 measurements gave
**178.591 ms median / 181.143 ms p95** on NVIDIA GeForce RTX 3060,
CUDA.jl 6.4.1, runtime 12.8 and ONNXRunTime.jl 1.4.0 (ORT 1.20.1).
Timing includes prepared CPU input transfer, inference, readout, CPU logits
return and synchronization. It excludes model loading (104.8 seconds in this
run), compilation, tokenization and `decide` answer calibration. No CPU affinity
was imposed. This measures the current fixed-shape exported graph; it is not
an optimized CUDA kernel benchmark. See the
[raw measurement JSON](assets/benchmarks/jeff-onnx-cuda-2026-10-02.json).
Export, validation and provider-placement logs and the measurement script are
retained under ignored `artifacts/cuda-validation/`.


## Historical Intel CPU measurements

<details>
<summary>Earlier Intel commits and experimental configurations</summary>

These commands use the removed environment switches from historical versions.

#### Intel CPU optimization trial (2026-10-01)

On an Intel
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

#### Portable activation and projection experiment

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

#### Forward thread policy and full-attention head experiment

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

#### Delta normalization and blockwise SiLU experiment

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

#### Portable CPU Delta projection workspace (Intel)

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

`JEFF_CPU_MLP_RESIDUAL_FUSION=1` additionally accumulates the MLP down
projection directly into the layer-owned residual using BLAS `beta=1`.
It is disabled by default and does not share mutable buffers between forwards.
On the same Intel/input/Float32/eight-worker/OpenBLAS-one-thread configuration,
30 forwards measured 294.338 ms and an independent 30-forward repeat measured
293.934 ms, versus a matched flag-off run of 297.079 ms. The roughly one-percent
timing difference is small; this is not a universal speedup claim.
Heap allocation decreased from 113,554,448 to 103,750,032 bytes per forward.
The maximum reference error was 1.2398e-5; alias, ownership, one/eight-worker,
full-suite and JET checks passed. The fixed Intel baseline speedup is about
1.69 times, still below the two-times goal. This option has not been benchmarked
on other platforms.

#### Optimization stopping checkpoint

The verified CPU implementation checkpoint is `7ccfe19`. On the Intel setup
above, the warmed median decreased from the fixed 495.254 ms baseline to
293.934 ms (about 1.69 times faster), and Julia heap allocation decreased from
392,752,624 to 103,750,032 bytes. The requested two-times target of 247.627 ms
was not reached. Optimization was stopped at the user's request.

A copy-free QKV/Z projection fusion trial was not adopted: matched 30-forward
medians were 293.576 ms without fusion and 291.372 ms with fusion; an independent
fusion repeat was 292.594 ms. The small difference came with increased loading
time (2.239 to 2.529 seconds) and process peak RSS (5.305 to 5.994 GB, including
loading and compilation). Numerical/ownership tests passed, but the initial
workspace type branching generated six JET runtime-dispatch reports. A revised
factory's benchmark completed; its full diagnostic revalidation was interrupted
when work stopped. The trial is absent from the verified implementation.

</details>

### Native CUDA measurements

On 2026-10-02, the optional CUDA extension loaded the original safetensors
checkpoint `mstrasser/Jeff-Qwen3.5-0.8B` at revision
`0f212b3e72acb4dde3f7da61e925d6ab7f819990`, without ONNX Runtime.
Hardware/software: RTX 3060, Julia 1.13.1, CUDA.jl 6.4.1, CUDA runtime 12.8,
cuBLAS 12.8.4, Float32 and the default CUDA math mode.

Thirty forwards after five warm-ups, batch 1 / length 256 / 101 active tokens,
measured median **70.4345 ms**, p95 **71.1010 ms**. This includes CPU input
upload, all layers, readout, CPU logits return and completion; model loading,
compilation, tokenization and answer calibration are excluded. All 255 output
columns of three independent PyTorch reference cases were validated, maximum
absolute error 8.5831e-6. [Full results](assets/benchmarks/jeff-native-cuda-2026-10-02.json).

`QDC_CUDA_TRIM_PADDING=1` skips leading masked positions, retaining interior
holes. The same input then computes 101 tokens: median **34.5000 ms**, p95
**35.0371 ms**, with the same validation protocol. This shorter workload is
reported separately. [Trimmed results](assets/benchmarks/jeff-native-cuda-trimmed-2026-10-02.json).

Every measured native forward allocated zero GPU buffers/bytes. Julia heap
allocation was typically 176,064 bytes full / 172,432 bytes trimmed; reported
GC time was zero in these samples. Reused workspace payload was 79,764,484
bytes full / 30,971,724 bytes trimmed, excluding model weights. These are
separate from CUDA pool retention: the full benchmark process reported
4.099 GiB active / 4.125 GiB reserved, including model/loading allocations.
Zero warm GPU allocation does not mean zero Julia heap allocation.

The initial generic CUDA implementation took about 1,014 ms and allocated
46.7 MB on the Julia heap per forward. Profile/Profile.Allocs identified
per-head/chunk arrays and repeated GEMMs. The extension uses recurrent delta
kernels, batched full attention, packed projections, fused normalization and
activation, cached cuBLAS coefficients and model-owned scratch. Scratch is
reused between layers with hidden-state ping-pong buffers and serialized
forwards; completion is awaited on success and exceptions. Multi-row delta
and launch/register configuration trials did not improve timings and were
not adopted.

`CUDA.@profile` on the final full forward captured 73.66 ms: GPU busy
68.55 ms, primary SGEMM 32.52 ms and delta recurrence 28.44 ms. This instrumented
trace is diagnostic, separate from the latency samples. In scripts, explicitly
`show(stdout, MIME"text/plain"(), profile_result)` to print CUDA.jl 6.4's report.
Host CUDA synchronization time overlaps device execution and must not be
added to kernel time. Profile logs remain under ignored `artifacts/cuda-validation/`.

GPU 0 performed benchmarks; GPU 1 performed correctness/type checks. The
optional tiny CUDA suite passed 184 checks, including mask/length changes,
GC reuse, retained CPU results, concurrent calls, failure recovery and device
restoration. Fifteen real-model reference cases passed 150 checks across both
trim settings, including lengths 1–512, multiple rows and interior mask holes.
The full CPU suite passed. Final `@code_warntype`/JET checks returned concrete
types with no optimization errors for logits and both attention layer types.
AllocCheck can flag host allocation paths, but cold scratch creation and CUDA
runtime paths produce findings; use warmed allocation measurements alongside
it rather than inferring GPU allocation counts from LLVM analysis.

Run these tools from an application environment containing CUDA and JeffClient:

```sh
julia --project=YOUR_ENV tools/benchmark_native_cuda.jl CHECKPOINT REFERENCE_JSON result.json 30 0
QDC_CUDA_TRIM_PADDING=1 julia --project=YOUR_ENV tools/benchmark_native_cuda.jl CHECKPOINT REFERENCE_JSON trimmed.json 30 0
julia --project=YOUR_ENV tools/verify_cuda.jl CHECKPOINT REFERENCE_JSON 1
julia --project=YOUR_ENV -e 'using CUDA; CUDA.device!(1); include("test/cuda.jl")'
```

The benchmark expects the reference JSON format produced by the project's
reference/export tools. `tools/build_cuda_reference.jl` regenerates the tiny
CUDA reference through PythonCall. cuBLAS coefficient reuse follows CUDA.jl
6.4.1's `gemmEx!` implementation; this internal API optimization was tested
with that version. Other CUDA 6 releases require revalidation.
