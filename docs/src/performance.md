# Measured inference speed

## Current results and remaining work

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
