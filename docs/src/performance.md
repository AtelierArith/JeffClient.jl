# Performance

## Measuring

```bash
julia --threads=8 --project=tools tools/benchmark.jl cuda   # or cpu, metal, amdgpu
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

## AMD GPU results (2026-10-06)

AMD Radeon 780M (gfx1103, 12 CUs), Ryzen 9 PRO 8945HS, Ubuntu 24.04, Julia
1.13.1, AMDGPU.jl 2.8.0, ROCm 7.0, Float32, batch 1, sequence length 256 with
101 active tokens. Leading padding is trimmed by default for AMDGPU, matching
the CPU policy, so both columns process 101 tokens.

| Workload | Median | p95 | Julia heap per forward |
| --- | ---: | ---: | ---: |
| AMDGPU, padding trimmed (101 tokens) | 191.1 ms | 199.0 ms | 0.7 MB |
| CPU, 16 Julia threads (101 tokens) | 210.1 ms | 218.6 ms | 25.3 MB |
| CPU, 8 Julia threads (101 tokens) | 207.7 ms | 229.1 ms | 23.9 MB |

The maximum logit error against the PyTorch reference was `1.2e-5` on AMDGPU
and `1.0e-5` on CPU. Reproduce with `julia --project=tools/amdgpu
tools/benchmark.jl amdgpu` and
`julia --threads=8/16 --project=tools tools/benchmark.jl cpu`; the machine was
idle. The iGPU shares system memory with the CPU, so the GPU advantage is
modest: rocBLAS FP32 on this part reaches only ~0.7 TFLOP/s (see below), and
storing projection weights transposed for contiguous `N,N` products is worth
roughly 12% over the transposed form. The GPU still uses far less Julia heap
per forward and no per-forward GPU allocation.

### rocBLAS is the AMDGPU ceiling

The AMDGPU forward is about 85% GEMM. On the Radeon 780M, rocBLAS FP32 tops
out near 0.7-0.8 TFLOP/s even for 8192³, below the same host's 16-thread
OpenBLAS (about 1.0 TFLOP/s at 4096³); for the N=101 shapes the model uses it
drops to ~0.3 TFLOP/s. rocBLAS FP16 is slower still (4.3 ms vs 1.8 ms for the
MLP gate/up shape) and its FP16 accumulation gives ~1e-2 logit error, so it is
unusable at the 2e-4 tolerance. The effective forward throughput (114 GFLOP in
191 ms) is therefore already near what rocBLAS allows; a further speedup would
need hand-written RDNA3 WMMA matrix-core kernels with FP16 inputs, FP32
accumulate and split-precision error compensation.

A discrete AMD GPU with dedicated VRAM and stronger rocBLAS tuning is expected
to show a much larger margin over the CPU.
