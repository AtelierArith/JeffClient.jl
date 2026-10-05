# Performance

## Measuring

```bash
julia --threads=8 --project=tools tools/benchmark.jl cuda   # or cpu, metal
QDC_CUDA_TRIM_PADDING=1 julia --threads=8 --project=tools tools/benchmark.jl cuda
julia --threads=8 --project=tools tools/benchmark.jl cpu --output result.json
```

The default input is `examples/data/parcel_reference.json`: Jeff-Qwen3.5-0.8B
at the pinned revision, batch 1, sequence length 256 with 101 active tokens,
Float32. The script checks the logits against the saved PyTorch reference,
runs 5 warm-up forwards and times 30. Each timed forward includes input
upload, the forward pass, readout, the CPU return of the scores and GPU
synchronization; checkpoint loading, compilation and tokenization are
excluded. Options such as `--checkpoint`, `--reference`, `--samples` and
`--gpu` are described at the top of the script.

`tools/benchmark_pytorch.jl` times the original Python implementation on the
same input through PythonCall (CPU or Apple MPS).

## Results (2026-10-05)

NVIDIA GeForce RTX 3060, Julia 1.13.1, CUDA.jl 6.4.2, CUDA runtime 13.4,
cuBLAS 13.8:

| Workload | Median | p95 | GPU allocation per forward | Julia heap per forward |
| --- | ---: | ---: | ---: | ---: |
| Full sequence (256 tokens) | 70.6 ms | 71.0 ms | 0 B | 184 KB |
| Leading padding trimmed (101 tokens) | 34.6 ms | 36.5 ms | 0 B | 178 KB |

The maximum logit error against PyTorch was `8.6e-6` (full) and `1.2e-5`
(trimmed). The trimmed run computes fewer tokens, so it is a different
workload. For comparison, an ONNX Runtime CUDA export of the same model took
a median 178.6 ms on this GPU (2026-10-02); that path has since been removed.

On the same Linux host's CPU (Intel Xeon E5-2699 v3, 8 Julia threads, default
policy) the median was 714 ms, but the host was shared (load average about 28
on 36 hardware threads) and p95 reached 1284 ms, so treat it as indicative
only.
