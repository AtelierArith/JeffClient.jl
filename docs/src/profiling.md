# Profiling and Metal tuning

This page preserves the allocation investigation and tuning history. For the
current comparison and its measurement conditions, see [Measured performance](performance.md).

### Type inference and allocation findings

`tools/inspect_native.jl` uses `@code_warntype`, JET 0.12.2 (available on this
Julia version), warmed `Profile.@profile`, and `Profile.Allocs.@profile`.
The initial layer vector erased its attention type parameter. This made
`hidden::Any` in the forward loop and caused nine JET runtime-dispatch reports
in JeffClient. Keeping the two concrete layer types in a small union removed
those reports. The hidden state now infers as `MtlMatrix{Float32}`; individual
full/DeltaNet layers, projections, and Metal matmul also have concrete outputs.
JET reporting is filtered to JeffClient and its Metal extension.

This type fix did **not** materially reduce allocation or latency: measured
Julia allocations were 2,845,990 before and 2,845,912 after, with heap bytes
145,771,392 versus 145,768,352. Median latency changed from 2.450 s to 2.492 s,
with overlapping ranges. It does not establish a speed improvement.

The 1% allocation profile sampled 28,483 allocations. The three Metal matmul
call variants accounted for 5,533 samples (~19%); repeatedly constructing and
uploading the DeltaNet lower-triangular mask accounted for 2,135 (~7.5%). Other
large sites were DeltaNet intermediate broadcasts, copies, and normalization.
These are sampled allocation counts, not time percentages or device byte usage.
The CPU profile includes repeated MPS/Objective-C submission, device-buffer
allocation, GC/memory-pressure handling, and final GPU waits. Readback stacks
include waiting for prior work; they do not isolate score-copy cost or GPU
kernel timings. Mask sharing, buffer reuse, batched command submission, and
shared uploads were then implemented. Recurrent DeltaNet, batched head products,
and fused convolution/normalization/softmax subsequently removed most of these
allocations. A later 10% sample found RMS normalization accounted for about 24%
of allocations before its fusion. RoPE/head fusion removed its slices and
layout copies. Remaining work includes fused residual/MLP operations,
explicit intermediate release, and reuse of workspace and MPS feed objects.
Detailed observations and intermediate measurements are in
[measurement notes](https://github.com/AtelierArith/JeffClient.jl/blob/main/memories/MEMORY.md).
An experimental task-local workspace is available with
`JEFF_METAL_WORKSPACE=1`. It keeps distinct arrays for each intermediate
allocation position and reuses their MPS tensor-data and feed/result value Vectors
after CPU score readback completes the GPU work. With the dedicated embedding
gather, paired Q/K normalization, fused delta gates, and dedicated mask/MLP/residual kernels, the prepared
batch-1/length-256 Float32 case measured 4,112 host allocations / 221,200 bytes
and a 193 ms median over
20 warmed runs. The preceding MLP optimization's 193 ms latency was also
reproduced in a separate process; residual addition reduced allocation further
without an established latency change.
Packing RMS parameters reduced the count by another 201 with unchanged bytes
and a 193 ms median.
The current path also fuses each layer's final residual sum into the following
input RMS and uses views for the last readout column; this removed another
471 allocations / 4,576 bytes without an established latency improvement.
Reusing the MPS command wrapper removed another 186 allocations / 5,952 bytes,
again with a 193 ms median.
Using the device arrays' dimensions instead of redundant scalar kernel
arguments removed another 78 allocations / 2,976 bytes without a latency gain.
Compiled kernel handle reuse removed another 120 allocations / 1,920 bytes,
also without an established latency gain.
Extending handle reuse to common RMS widths and attention helpers removed
another 999 allocations / 47,024 bytes without an established latency gain.
Specializing Q/K, recurrent, and RoPE launches and reusing tensor-data for
past workspace slots removed another 871 allocations / 40,544 bytes. DeltaNet
gates now write directly into matrix outputs, avoiding reshape wrappers; the
warmed workspace forward allocates no new MPS tensor-data wrappers in the
full allocation profile.
It retains 447 arrays / 857,899,008
device-buffer bytes, 296 tensor-data objects, and 199 reusable feed Vectors;
these bytes are separate from
the free pool and exclude weights and native MPS resources. This option remains
disabled by default while its lifetime and memory behavior are evaluated.
To release the current task's workspace references after using it:

```julia
Base.get_extension(JeffClient, :JeffClientMetalExt).clear_forward_workspace!()
```

This waits for GPU completion and drops cached references. Buffer return still
uses normal ownership and GC; it does not immediately free all resident memory.

An additional experiment, `JEFF_METAL_PACKED_MLP=1`, packs gate/up weights when
loading a Metal backend. It reduces each MLP to two matrix products and a gate
kernel; the backend retains packed weights instead of the original pair.
With workspace enabled, B1/L256 measured 4,001 allocations / 203,760 bytes and
a 192.5 ms median, versus 4,112 / 221,200 and 193.4 ms without packing.
Workspace buffer retention increases from 857,899,008 to 945,979,392 bytes.
B2/L512 measured 771.8 versus 776.1 ms, with 176,160,768 additional retained
buffer bytes. These small latency differences overlap the measurement ranges;
packing is disabled by default. It trades larger temporary buffers and loading
copies for fewer matrix submissions and host allocations. Set the variable
before loading the backend. `tools/benchmark_stages.jl` compares both projection
strategies using an unpacked backend, so run it with `JEFF_METAL_PACKED_MLP=0`.
Without workspace, packed MLP measured 8,191 allocations / 342,080 bytes /
194.1 ms, versus the default 8,446 / 364,512 / 195.2 ms; latency ranges overlap.

Two further options, both disabled by default, reduce work on left-padded inputs:

- `JEFF_METAL_TRIM_PADDING=1` skips only the leading zero-mask prefix; interior
  mask holes remain. B1/L256 with 101 active tokens and workspace enabled measured
  86.1 ms / 4,111 allocations / 217,584 host bytes over 20 samples, versus
  193.4 ms without trimming. The computed sequence length becomes 101.
- `JEFF_METAL_SHAPE_WORKSPACES=1`, together with `JEFF_METAL_WORKSPACE=1`,
  retains up to two workspaces by computed sequence length. With trimming,
  B2/L512 with active lengths 512/256 measured 574.8 ms median / 577.4 ms p95 /
  579.7 ms maximum over 20 samples, and 8,334 allocations / 749,024 host bytes.
  The two workspaces retain 2,623,995,904 device-buffer bytes. A single workspace
  repeatedly replacing shapes measured 17,114 allocations and a 9.2-second
  maximum. Longer measurements are needed to establish tail behavior.

RoPE tables now retain two recent configurations per queue. After this change,
the same B2 case over 50 samples measured 8,274 allocations / 448,608 host bytes,
581 ms median / 590 ms p95 / 591 ms maximum. This reduces table construction
allocations; the measurements do not establish an additional latency improvement.

Shape workspaces use LRU eviction and, after completion, evict older entries if
their combined buffer bytes exceed one quarter of the recommended working set.
A single current workspace can exceed that limit. Weights, free pools, and peak
memory are excluded. The clear function above releases all shape workspaces.
Independent PyTorch references pass for 15 cases, including interior mask holes;
shape-workspace ownership checks pass. Its six inspected JET targets report no
errors, and allocation profiling records no new MPS tensor-data wrappers.

`JEFF_METAL_FUSED_DELTA_MASK=1` is another experiment, disabled by default. It
applies DeltaNet's input mask while writing the RMS result, removing a separate
kernel and intermediate array. All-one prepared masks skip multiplication.
Full attention and residual values keep their existing mask behavior.
With trimming and shape workspaces enabled, B2/L512 measured 7,480 allocations /
424,544 host bytes versus 7,866 / 435,552 without fusion. Retained workspace
buffers decrease by 56,623,104 bytes to 2,567,372,800 bytes. Medians around
580–594 ms overlap the non-fused control; this is an allocation and buffer
retention improvement, without a confirmed latency improvement. Independent
15-case references, mask-switching slot identity/GC checks, and the six JET
targets pass. Allocation profiling records 376 rather than 412 KernelState
objects for B2, consistent with removing 36 launches.

`JEFF_METAL_BATCHED=1` enables experimental joint GPU execution for multiple
samples, disabled by default. Convolution, recurrent state, attention masks and
RoPE positions remain separate per sample. It uses a shared padded length and a
single workspace; the shape-workspace bank and fused DeltaNet mask setting
currently apply to the row execution path. B1 and unsupported widths use row
execution. Initial B2/L512 measurements reduce host allocations to 4,152 but
take 792 ms, compared with approximately 586 ms for individually trimmed rows.
The batch computes lengths 512/512 versus 512/256 and retains about 3.53 GB of
workspace buffers. This experiment is not a faster replacement for mixed-length
row execution. For equal-length B2/L1, 50-run medians are 22.8 ms batched versus
43.9 ms row-wise, with 3,947 versus 7,751 host allocations. This short-input
result does not establish a speedup for longer or differently padded samples.
Independent 15-case references pass; further performance and
configuration validation is in progress.

The row-path type check reports no JET errors for the six inspected JeffClient
and Metal-extension targets; normalization's Bool dispatch is explicitly split.

```bash
julia --project=tools tools/inspect_native.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx/reference.json
# Type reports only:
JEFF_INSPECT_PROFILE=0 julia --project=tools tools/inspect_native.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx/reference.json
# Increase allocation sampling when the optimized forward allocates less:
JEFF_ALLOC_SAMPLE_RATE=0.1 julia --project=tools tools/inspect_native.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx/reference.json
# Inspect a selected case in an expanded reference document:
julia --project=tools tools/inspect_native.jl CHECKPOINT_DIRECTORY artifacts/metal-validation/reference-with-mask-holes.json 12
```

AllocCheck could not be combined with this Metal environment: its registered
versions require GPUCompiler ≤1.23, while Metal 1.11 requires GPUCompiler ≥2.7.
The runtime allocation measurements above use Julia's built-in allocation profiler.

### Cthulhu and TypedSyntax

`tools/inspect_typed_source.jl` displays inferred types on source with TypedSyntax,
prints raw inferred IR, or starts Cthulhu's interactive descent. Cthulhu 3.0.2 and
TypedSyntax 1.5.4 were used on Julia 1.13.1. They are tools dependencies.

Following the allocation profile into `MPSGraphTensorData`, Cthulhu exposed the
shape conversion `NSArray(NSNumber.(collect(tuple)))`. This concrete-typed path
still built containers per call. Keeping the immutable shapes in the cached graph
reduced forward allocations by 14.4%. The unmanaged `NSArray` objects need explicit
retain/release to survive autorelease-pool drainage; GC/repeated-forward validation
checks that lifetime. Tensor-data wrappers and feed dictionaries still allocate.

```bash
# Batch source report; accepts old and expanded reference JSON:
julia --project=tools tools/inspect_typed_source.jl CHECKPOINT_DIRECTORY REFERENCE_JSON all source
# Raw inferred IR:
julia --project=tools tools/inspect_typed_source.jl CHECKPOINT_DIRECTORY REFERENCE_JSON submit typed
# Explicit interactive mode, in a terminal:
julia --project=tools tools/inspect_typed_source.jl CHECKPOINT_DIRECTORY REFERENCE_JSON tensor descend
```

Targets include `logits`, `full`, `delta`, `rope`, `submit`, `tensor`, and
`cached-tensor`. Macro/closure source mapping can omit or misplace annotations;
check surprising results against raw IR or Cthulhu's typed view. Allocation and
GPU timing still require runtime measurements. See the
[Cthulhu](https://github.com/JuliaDebug/Cthulhu.jl) and
[TypedSyntax](https://github.com/JuliaDebug/Cthulhu.jl/blob/master/TypedSyntax/README.md)
documentation for their roles and mapping limitations.
