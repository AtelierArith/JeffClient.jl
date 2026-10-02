# Inference and accelerators

## Setup

Install Julia 1.13 and Git, then run:

```bash
git clone https://github.com/AtelierArith/JeffClient.jl.git
cd JeffClient.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project examples/native_inference.jl
```

Internet access is needed for the initial Julia dependency installation and
checkpoint download. The native demo requires no Python setup or ONNX export.

For Metal examples, on a Mac with an Apple GPU, first install the tools
environment from the repository root:

```bash
julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
```

## Real-checkpoint demo

Run the bundled parcel-classification example from the repository root:

```bash
julia --project examples/native_inference.jl
# Or reuse a local checkpoint:
julia --project examples/native_inference.jl CHECKPOINT_DIRECTORY cpu
# Apple GPU, using the tools environment that includes Metal:
julia --project=tools examples/metal_inference.jl
# Or reuse local weights on Metal:
julia --project=tools examples/metal_inference.jl CHECKPOINT_DIRECTORY
```

The default command resolves the pinned Jeff-Qwen3.5-0.8B checkpoint, downloading
roughly 1.7 GB only when it is not cached. The example calls `decide` with a real,
pre-tokenized prompt in `examples/data/parcel.json`. It runs model inference and
probability calibration in Julia, without Python or an ONNX export.

The CPU example was executed with the actual model weights:

```text
Device: cpu
Input: The parcel arrived crushed and the customer wants a replacement.
Question: Which team should handle this?
Choice: delivery
  refund: 0.003434
  delivery: 0.996566
Confidence: 0.993133
```

The fixture's tokens were prepared using the original Jeff tokenizer for revision
`0f212b3e72acb4dde3f7da61e925d6ab7f819990`. Its option order is refund, then delivery.
The displayed state is informational: changing it or the question does not
regenerate tokens. General text tokenization remains outside this API.

The standalone Metal example imports Metal, checks `Metal.functional()`, disables
scalar GPU indexing, constructs `NativeBackend(checkpoint; device=:metal)`, and
returns the decision to CPU. It uses the same real weights and prepared prompt
as the CPU example. It requires an Apple GPU; no Python or ONNX export is used.

## Supplement: ONNX Runtime

To try the ONNX interface without downloading Jeff's weights, run:

```bash
julia --project examples/prepared_inference.jl
```

The `test/fixtures/logits.onnx` file is tracked by Git and included in the
clone. The example does not generate it. This tiny Identity graph processes
supplied scores; it contains no Jeff weights and prints `Choice: yes` with
probability `0.75`. No Python or checkpoint download is required.

### Prepared-tensor inference

Your graph must output raw option logits of shape `(batch, options)`. Pass
arrays in ONNX logical dimension order; do not reverse dimensions manually.
Set the output name, temperature, and option limit to match your export.

```julia
using JeffClient

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
ONNXBackend("jeff.onnx";
    output_name="logits", temperature=1.0, max_options=254) do backend
    result = decide(backend, inputs, question)
    println(result.choice, " ", result.probabilities)
end
```

The `do` block returns its last expression and releases the ONNX session on
both normal completion and exceptions. For a batch, pass an ordered vector of questions corresponding to the output
rows. Yes/no questions use false then true columns. Score questions return an
expected value on a zero-based scale. Choice confidence follows Jeff's formula
and is not the same as the winning probability.

### CUDA through ONNX Runtime

Install `CUDA` and `cuDNN` in your application environment, import both before
constructing `ONNXBackend(...; execution_provider=:cuda)`. Compatibility depends
on ONNXRunTime's CUDA runtime requirements and the graph's operators. Other
providers are not currently exposed by this package.

Native Julia CUDA inference is available separately through
`NativeBackend(...; device=:cuda)` after importing CUDA. It loads safetensors
directly and does not execute an ONNX graph.
ONNXRunTime.jl 1.4.0 requires a functional CUDA 12.x runtime. If CUDA selects
13.x, run `CUDA.set_runtime_version!(v"12.8")` in your application environment
and restart Julia. On 2026-10-02, Linux CUDA inference passed with CUDA.jl 6.4.1
and runtime 12.8: a MatMul fixture ran on the CUDA provider and matched Julia's
matrix product, including answer generation through `decide`. The full Jeff
0.8B checkpoint also passed three reference cases and ran in a median 178.6 ms
on RTX 3060 (Float32, batch 1, sequence length 256). Results and environment conditions
are recorded in [Profiling and measurements](profiling.md#CUDA-/-ONNX-Runtime-verification).

## Native Julia model and Metal

For NVIDIA GPUs, install CUDA in your application environment and use:

```julia
using JeffClient, CUDA
CUDA.allowscalar(false)
CUDA.device!(0)
backend = NativeBackend(checkpoint; device=:cuda)
scores = logits(backend, inputs)
```

This optional extension uses Julia CUDA kernels and cuBLAS, with Float32
weights and outputs. It processes the complete prepared sequence, including
left padding. Scratch buffers belong to each model and are reused only after
its CUDA stream completes; calls sharing a model are serialized. Returned
scores are owned CPU arrays. A model remains usable if the caller changes
the active CUDA device; the forward temporarily selects the model's device.
Native CUDA does not require cuDNN or an ONNX export.

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
projections, and row-wise batch execution by default (experimental joint Metal batching is available). Tokenization, image
inputs, generation/KV caching, and training are pending. Performance tuning is ongoing.
Total device memory usage has not been measured.

Set `ENV["JEFF_CUDA_TRIM_PADDING"] = "1"` to skip leading masked tokens.
This option preserves interior mask holes and is disabled by default so full
sequence benchmarks remain comparable. See [performance](performance.md) for
full-sequence and trimmed measurements.
