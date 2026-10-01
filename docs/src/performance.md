# Measured inference speed

## Automatic CPU configuration

CPU tuning environment variables have been removed. Normal CPU execution
automatically uses the verified portable configuration on Intel/other platforms,
or the previously fastest Accelerate configuration on Apple Silicon macOS
when Accelerate BLAS forwarding is available. SIMD domain/alias guards and
forward-local ownership are retained; the unadopted QKV/Z fusion is not enabled.
These policies select measured configurations, not a guarantee of the fastest
implementation for every CPU or input.

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

On the Intel i9-9900K / Julia 1.13.1 / Float32 / parcel input below, 30 warmed
forwards with no CPU tuning environment variables measured 293.433 ms median
and 316.923 ms p95, with OpenBLAS one thread and eight Julia workers.
Heap allocation was 103,849,264 bytes and maximum absolute reference error
was 1.2398e-5. Full tests, one/eight-worker numerical and ownership checks,
and JET checks passed. Linux measurements of the automatic configuration are
recorded in [Profiling and measurements](profiling.md). The automatic Apple M4 configuration has not been remeasured;
that platform choice is based on the historical results on that page.

## Measurements

See [Profiling and measurements](profiling.md) for the Linux CPU results,
Python comparison, thread configurations, Octavian experiment and historical
CPU/Metal trials. Raw benchmark JSON is linked beside each measurement.
