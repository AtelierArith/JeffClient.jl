# JeffClient.jl

Julia inference infrastructure for [Jeff](https://github.com/firelex/jeff).
The implementation runs **ONNX models and a native Julia Qwen3.5 model with
prepared input tensors**. Jeff-Qwen3.5-0.8B has been compared with PyTorch on
ONNX CPU, native Julia CPU, and native Julia Metal. Text tokenization and
additional accelerators are planned in [PLAN.md](PLAN.md).

Run `julia --project examples/prepared_inference.jl` for a working CPU demo
using the tiny test graph. It demonstrates the inference boundary and probability
calibration; it does not classify text or use Jeff's trained weights.

The Julia dependency is named `ONNXRunTime` (capital R and T). Python is not
required by this inference API. ONNX Runtime itself is a native library.

## Model downloads and cache

```julia
using JeffClient

checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990")
# Reuse locally, with network access disabled:
checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990", offline=true)
```

This downloads the **source checkpoint**, not an ONNX model. The loader checks
the existing Hugging Face cache, then JeffClient's Scratch.jl cache under your
Julia depot. Only model/config/tokenizer/license files are fetched; videos and
assets are skipped. Supply `cache_dir="..."` to choose another cache root.
`HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`, and `HF_HUB_OFFLINE` are
supported. Local checkpoint directories can also be passed directly.

## Export and validate Jeff

The one-time export uses the Python environment for `extern/jeff` plus
`tools/requirements.txt`, accessed from Julia through PythonCall.jl. Download the source weights using Julia above, then
pass the returned checkpoint directory to the exporter:

```bash
uv pip install --python extern/jeff/.venv/bin/python -r tools/requirements.txt
julia --project=tools tools/export_onnx.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx \
    --sequence-length 256 --source-revision 0f212b3e72acb4dde3f7da61e925d6ab7f819990
julia --project examples/verify_export.jl artifacts/jeff-0.8b-onnx
```

Export output directories must be new. The initial graph uses a fixed batch
size (default 1) and sequence length (default 256); inputs must be left-padded
without truncating. The float32 graph/weights take about 3 GB and are gitignored.
The verification example compares Julia's ONNX scores with three recorded
PyTorch references, including Japanese text. It consumes prepared token IDs,
so Python is not involved in verification or inference.

Use `backend = load_export("artifacts/jeff-0.8b-onnx")` to load the graph with
the checkpoint's temperature and option limit automatically. Call `close(backend)`
after use.

The reference source checkout is intentionally ignored by Git. To set it up,
run `git clone https://github.com/firelex/jeff extern/jeff`, then create its Python
environment according to its README. `tools/export_onnx.jl` uses PythonCall.jl
with that environment (`JULIA_PYTHONCALL_EXE` overrides the interpreter).
PythonCall is a tools dependency, not a dependency of Julia model inference.

## Prepared-tensor inference

Your graph must output raw option logits of shape `(batch, options)`. Pass
arrays in ONNX logical dimension order; do not reverse dimensions manually.
Set the output name, temperature, and option limit to match your export.

```julia
using JeffClient

backend = ONNXBackend("jeff.onnx";
    output_name="logits", temperature=1.0, max_options=254)
question = ChoiceQuestion([
    "refund" => "Refunds and payments",
    "delivery" => "Damaged or lost parcels",
])

# Tokens must already encode the checkpoint's exact prompt and choice order.
# Example dimensions only; use the actual tensors and inputs your graph requires.
inputs = Dict(
    "input_ids" => reshape(Int64[101, 102, 103], 1, 3),
    "attention_mask" => ones(Int64, 1, 3),
)
try
    result = decide(backend, inputs, question)
    println(result.choice, " ", result.probabilities)
finally
    close(backend)
end
```

For a batch, pass an ordered vector of questions corresponding to the output
rows. Yes/no questions use false then true columns. Score questions return an
expected value on a zero-based scale. Choice confidence follows Jeff's formula
and is not the same as the winning probability.

## CUDA

Install `CUDA` and `cuDNN` in your application environment, import both before
constructing `ONNXBackend(...; execution_provider=:cuda)`. Compatibility depends
on ONNXRunTime's CUDA runtime requirements and the graph's operators. Other
providers are not currently exposed by this package.

## Native Julia model and Metal

The text-only Qwen3.5 forward pass loads source safetensors directly: embeddings,
partial RoPE, grouped full attention, Gated DeltaNet, RMS normalization, SiLU MLP,
and Jeff's readout are implemented in Julia. No ONNX export or Python is needed
for this inference path.

```julia
using JeffClient

checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990")
backend = NativeBackend(checkpoint) # CPU
# Use the same prepared input_ids, attention_mask, and question as above.
result = decide(backend, inputs, question)
```

For Apple GPU execution, install and import Metal in your application environment:

```julia
import Metal
backend = NativeBackend(checkpoint; device=:metal)
result = decide(backend, inputs, question)
```

Metal is an optional package extension. It uses Metal.jl arrays, cached MPSGraph
products encoded into the current command batch, pooled output buffers, shared
uploads, and Julia Metal kernels for recurrent DeltaNet, depthwise convolution,
RMS/L2 normalization, causal masked softmax, and fused partial RoPE/head layouts
with queue-local trigonometric tables. Full-attention heads use batched matrix
products. Graphs also retain immutable Objective-C shape metadata for repeated
tensor-data construction. Product-only graphs omit `beta*C`, so recycled destination
contents are never read. Fixed feed/result keys are also retained with the graph;
tensor dictionaries are built from their changing values without Julia Dict
storage or keys/values conversion copies. NaN-poisoned destinations are checked
by a verifier.
For DeltaNet key widths above 256, the stable chunked triangular solve remains
available. Forming its inverse with repeated products was numerically unstable.
This backend uses Metal.jl directly and does not depend on MLX.
The buffer/command implementation is adapted from Laya.jl. It uses Metal
internal APIs, so compatibility is pinned to the verified Metal 1.11.1.

Three real English/Japanese cases agreed with PyTorch within `1.72e-5` on native
CPU (maximum absolute logit error), using prepared sequences of length 256.

A broader run on Apple M4, Julia 1.13.1, and Metal 1.11.1 compared 12 cases
three times with independent PyTorch float32 scores: mixed English/Japanese prompts
(batch 3), and padded/unpadded pairs of lengths 1, 63, 64, 65, 127, 128, 129,
255, 256, 257, and 512. The current fused implementation's maximum absolute logit
error was `3.361702e-5`.
Scalar GPU indexing was disabled, and GC ran between cases. Random-token cases
probe numerical agreement rather than the quality of language classification.

```bash
julia --project=tools tools/verify_native.jl CHECKPOINT_DIRECTORY cpu
julia --project=tools tools/verify_native.jl CHECKPOINT_DIRECTORY metal
# Generate independent references through PythonCall, then validate on Metal:
julia --project=tools tools/build_metal_reference.jl CHECKPOINT_DIRECTORY artifacts/metal-validation/reference.json
julia --project=tools tools/verify_metal.jl CHECKPOINT_DIRECTORY artifacts/metal-validation/reference.json 2
# NaN product destinations, real RMS widths, RoPE/grouped layouts and gates:
julia --project=tools tools/verify_metal_primitives.jl
```

Current limits: float32 text inference, default partial RoPE, bias-free
projections, and one sequence at a time within a batch. Tokenization, image
inputs, generation/KV caching, and training are pending. Performance tuning is ongoing.
Total device memory usage has not been measured.

### Measured inference speed

Apple M4, Julia 1.13.1, Metal 1.11.1, PyTorch 2.14.0; pinned Jeff 0.8B,
batch 1, sequence length 256 (101 active tokens), same prepared English prompt,
20 warmed forward passes for current Julia Metal and PyTorch MPS Float32;
the earlier CPU and BF16 runs used 10. Backends were measured separately. Inputs are
prepared before timing. Loading/tokenization are excluded; readout and returning
scores to CPU are included. GPU measurements wait for completion.

| Backend | Weight precision | Samples | Median per forward |
| --- | --- | ---: | ---: |
| Native Julia CPU (8 BLAS threads) | Float32 | 10 | 1.834 s |
| Native Julia Metal (fused kernels) | Float32 | 20 | 0.214 s |
| Original Jeff / PyTorch CPU (8 threads) | Float32 | 10 | 2.988 s |
| Original Jeff / PyTorch MPS | BF16 (original default) | 10 | 0.333 s |
| Original Jeff / PyTorch MPS | Float32 | 20 | 0.350 s |

For this prepared batch-1 case, Julia Metal takes about **38% less time than
PyTorch MPS at the same Float32 precision** (~1.60× throughput). This comparison
does not establish performance for other lengths or batch sizes.
The Python runs call the original Jeff `forward` with its backbone wrapper,
readout, and option masking. They use the installed reference DeltaNet and
convolution implementations, without Flash Linear Attention or causal-conv1d.

Julia Metal's model loading took 1.62 s and its first forward took 13.01 s
(including compilation, excluding package imports); subsequent timing is above.
BenchmarkTools measured 21,943 Julia heap allocations / 1.03 MB per
Metal forward; this does not measure GPU buffer bytes. Host profiling includes
MPS submission, Objective-C calls, array allocation, and synchronization.

Following the Laya-based buffer changes and kernel fusion, median Metal latency
fell from 2.492 s to 0.214 s (~11.6× faster). Allocation counts fell from 2,845,912
to 21,943 (~99.2%), and Julia heap bytes from 145,768,352 to 1,027,552 (~99.3%). Device-only buffers
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
The latest median was 214 ms;
small timing differences require repeat measurements.

The trial recorded 7,072 private-buffer reuses and 425 shared-upload reuses.
It retained about 1.75 GB of free device buffers afterward: reducing allocation
uses a cache and does not imply lower total resident memory. Private free caches
are trimmed on allocation pressure at one quarter of the recommended working
set per queue; retained shared uploads are limited to 64 MB per queue. The pool
size depends on GC and prior calls; it is not a measurement of peak GPU memory.

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
[memories/MEMORY.md](memories/MEMORY.md).
The latest type check reports no JET errors for the six inspected JeffClient
and Metal-extension targets; normalization's Bool dispatch is explicitly split.

```bash
julia --project=tools tools/inspect_native.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx/reference.json
# Type reports only:
JEFF_INSPECT_PROFILE=0 julia --project=tools tools/inspect_native.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx/reference.json
# Increase allocation sampling when the optimized forward allocates less:
JEFF_ALLOC_SAMPLE_RATE=0.1 julia --project=tools tools/inspect_native.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx/reference.json
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

## Development

Julia 1.13 is the development baseline. Run `julia --project -e 'using Pkg;
Pkg.instantiate(); Pkg.test()'`. CPU tests use a tiny ONNX graph and a small
synthetic Qwen fixture with independent PyTorch reference scores. They do not
download Jeff's weights or require Python.
# JeffClient.jl
