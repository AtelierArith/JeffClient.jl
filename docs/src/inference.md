# Inference and accelerators

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
projections, and row-wise batch execution by default (experimental joint Metal batching is available). Tokenization, image
inputs, generation/KV caching, and training are pending. Performance tuning is ongoing.
Total device memory usage has not been measured.

