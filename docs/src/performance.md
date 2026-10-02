# Measured inference speed

## Automatic CPU configuration

CPU tuning environment variables have been removed. Normal CPU execution
automatically uses the verified portable configuration on Intel/other platforms,
or the Accelerate configuration on Apple Silicon macOS
when Accelerate BLAS forwarding is available. SIMD domain/alias guards and
forward-local ownership are retained; the unadopted QKV/Z fusion is not enabled.
These policies do not guarantee the fastest implementation for every CPU or input.

JeffClient initializes the process-wide BLAS thread count on import: one thread
for portable execution with multiple Julia workers, up to eight for single-worker
execution, and Accelerate-managed threading on Apple Silicon. This also affects
other BLAS users in the process. Do not change BLAS configuration during inference.
Julia's worker pool is selected at startup, not resized by this library.

```sh
julia --threads=8 --project=tools examples/native_inference.jl
julia --threads=8 --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu examples/data/parcel_reference.json 1 30 artifacts/benchmarks/cpu-defaults.json
```

The benchmark records the selected policy and actual BLAS thread configuration.
Internal task-local diagnostic scopes can still compare reference algorithms;
they are not a supported configuration API.

## Linux: matched CPU comparison (2026-10-01)

`./tools/linux-cpu.sh` runs Python/PyTorch and Julia sequentially in fresh
processes, alternating their order across repeats. `--threads 1` gives a strict
single-thread comparison; the default `--threads 8` gives both implementations
the same eight physical cores. Julia uses eight workers and one BLAS thread;
PyTorch uses eight intra-op threads and one inter-op thread. Extra Julia
interactive workers are disabled. Affinity constrains the CPU budget but does
not isolate those cores from other host workloads.

The current matched trial used an Intel Xeon E5-2699 v3, Linux x86_64,
Julia 1.13.1, Python 3.12.3, PyTorch 2.14.0+cu130 and Transformers 5.17.0.
Julia used optional MKL.jl 0.9.1 / MKL 2025.2; PyTorch reports MKL 2024.2.
OpenBLAS remains the normal Julia backend; this table is an explicit MKL trial.
Both run the same Jeff 0.8B checkpoint, revision
`0f212b3e72acb4dde3f7da61e925d6ab7f819990`, in Float32, parcel case 1,
batch 1, all 256 positions (101 active). Julia padding trim, final-query-only,
final-token-only, recurrent Delta and Octavian are disabled; chunk size is 64.
Readout and CPU score return are included. Loading, compilation, tokenization
and validation are excluded. Both collect garbage after cold validation, then warm up ten forwards and
measure 30 forwards. Automatic GC stays enabled; there is no collection forced
between samples. Three fresh processes per implementation ran in alternating order.

All **255 readout logits** passed the independent saved PyTorch reference
with atol/rtol `2e-4`; maximum error was `1.04904e-5` in both implementations.
The checkpoint's `max_options=254` is a supported question limit, not the
readout width: the Linux adapter requests `model.readout.out_features` to
validate the last column as well.

| Implementation | Run | Median (ms) | p95 (ms) | Python / Julia median |
|---|---:|---:|---:|---:|
| Python / PyTorch | 1 | 812.884 | 813.567 | — |
| Julia / MKL | 1 | 767.658 | 789.467 | 1.0589× |
| Python / PyTorch | 2 | 809.849 | 812.309 | — |
| Julia / MKL | 2 | 771.419 | 795.555 | 1.0498× |
| Python / PyTorch | 3 | 805.778 | 807.168 | — |
| Julia / MKL | 3 | 795.975 | 824.219 | 1.0123× |

**Julia's median is faster in all three paired MKL runs**, by 1.2–5.6% in
elapsed time (speedup 1.01–1.06×). Its p95 is lower in two runs and higher in
the third; the fastest median does not establish a tail-latency guarantee.
The host was not exclusively isolated. See the
[validated timings, GC series, source/model hashes and conditions](assets/benchmarks/linux-cpu-2026-10-01-matched-mkl.json).

The normal OpenBLAS 0.3.30 backend was measured separately with the same
30-sample/10-warm-up protocol and two fresh processes per implementation:

| Run | Python median / p95 (ms) | Julia median / p95 (ms) | Python / Julia median |
|---|---:|---:|---:|
| 1 | 806.361 / 808.247 | 848.217 / 867.578 | 0.9507× |
| 2 | 809.923 / 814.286 | 841.139 / 874.879 | 0.9629× |

Default Julia/OpenBLAS is still 3.9–5.2% slower by median on this host/input.
The MKL result therefore does not establish a default-backend win.
[OpenBLAS timings and conditions](assets/benchmarks/linux-cpu-2026-10-01-matched-openblas.json)
retain the full output guards, GC series and provenance.

To reproduce with a local checkpoint:

```sh
# Normal backend: OpenBLAS; use --threads 1 for the strict single-thread case.
./tools/linux-cpu.sh --checkpoint "$CHECKPOINT" --threads 8 --samples 30 --repeats 2 --warmups 10

# Optional MKL trial: prepare a separate environment before measurement.
julia --startup-file=no --project=artifacts/benchmarks/mkl-env -e 'using Pkg; Pkg.add(PackageSpec(name="MKL", version="0.9.1"))'
./tools/linux-cpu.sh --checkpoint "$CHECKPOINT" --threads 8 --cpus 0-7 --samples 30 --repeats 3 --warmups 10 --mkl-project artifacts/benchmarks/mkl-env
```

Omit `--cpus` to select distinct physical cores from the process's allowed CPU
set automatically. The measured invocation used `--cpus 0-7`, the checkpoint's
local snapshot path, and the separately prepared MKL environment under
`artifacts/cpu-python-gap/mkl-env`. Use `--setup-python` if the project's uv
environment has not been prepared. JSON/logs and a validated summary are saved
under a new ignored `artifacts/benchmarks/` directory. Failed, partial, or
mismatched runs are rejected before summary generation.

The earlier Python advantage came mainly from compiled CPU kernels: PyTorch's
profile spent 68.5% in matrix multiplication. Julia's scalar activation
fallbacks, strided Delta head sequences and repeated attention/normalization
arrays added work. The implementation now uses guarded SIMD activation,
contiguous Delta head storage, in-place softmax/RMS and forward-owned buffers
for full attention and residual normalization. Each worker owns its attention
scratch; RoPE tables are computed once per forward.

In the allocation investigation, BenchmarkTools' per-forward Julia heap estimate
fell from **324.4 MB to 58.7 MB** (81.9%). This is allocation traffic, not retained model
memory or peak RSS. RMS/full-attention reuse alone barely changed the median;
the additional in-place Delta RMS reduced allocation and improved the measured
median. The final Julia runs had median reported GC time zero and maximum
12.6–16.9 ms; their slowest samples had no reported GC time. Allocation and GC
explained part of the earlier tail, but do not explain all remaining variation.
See [Profiling and measurements](profiling.md) for the investigation and checks.

## Apple Silicon measurements

On Apple Silicon macOS, use `./tools/mac-M-series.sh` for the matched CPU and
PyTorch MPS / Julia Metal.jl comparisons. The default includes both the
one-thread CPU comparison and PyTorch 8 / Julia 8 with automatic Accelerate,
reported separately because their thread budgets differ. It uses Float32/full
sequences, CPU input upload for GPU, and two independent runs. `--mlx` adds
the explicitly adapted MLX results. See `--help` for reproducible output paths.

See [Profiling and measurements](profiling.md) for the new Apple M4 and Linux CPU results,
Python comparison, thread configurations, Octavian experiment and historical
Intel CPU trials. Raw benchmark JSON is linked beside each measurement.

## NVIDIA CUDA measurements

Jeff-Qwen3.5-0.8B on an RTX 3060, Float32, batch 1, input length 256
(101 active tokens), 30 measured forwards after five warm-ups:

| Backend | Median | p95 |
| --- | ---: | ---: |
| ONNX Runtime CUDA | 178.59 ms | 181.14 ms |
| NativeBackend CUDA, full sequence | 70.43 ms | 71.10 ms |
| NativeBackend CUDA, leading padding trimmed | 34.50 ms | 35.04 ms |

Full-sequence native inference is 2.54 times faster than this ONNX export.
The trimmed option computes 101 tokens, so it is a separate workload. Timing
includes input upload, readout, CPU score return and synchronization; loading,
compilation and tokenization are excluded. Native inference allocates no GPU
buffers after warm-up on this input, but still allocates about 176 KB on the
Julia heap per full forward. See [profiling](profiling.md#Native-CUDA-measurements)
for raw results, validation, memory accounting and reproduction commands.
