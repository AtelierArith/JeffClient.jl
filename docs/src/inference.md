# Inference on CPU, CUDA and Metal

## Demo

```bash
julia --threads=8 --project examples/native_inference.jl
# Or reuse a local checkpoint:
julia --threads=8 --project examples/native_inference.jl CHECKPOINT_DIRECTORY
```

The first run downloads the pinned Jeff-Qwen3.5-0.8B checkpoint (about 1.7 GB).
The example calls `decide` with a real pre-tokenized prompt from
`examples/data/parcel.json`:

```text
Device: cpu
Input: The parcel arrived crushed and the customer wants a replacement.
Question: Which team should handle this?
Choice: delivery
  refund: 0.003434
  delivery: 0.996566
Confidence: 0.993133
```

The tokens were prepared with the original Jeff tokenizer for revision
`0f212b3e72acb4dde3f7da61e925d6ab7f819990`, with the options in the order
refund, delivery. Editing the displayed text does not regenerate them.

## API

```julia
using JeffClient

backend = NativeBackend("mstrasser/Jeff-Qwen3.5-0.8B") # or a local directory
question = ChoiceQuestion([
    "refund" => "Refunds and payments",
    "delivery" => "Damaged or lost parcels",
])
# (batch, sequence) token ids and mask, left-padded; the tokens must encode
# the checkpoint's prompt and option order.
inputs = Dict("input_ids" => input_ids, "attention_mask" => attention_mask)
result = decide(backend, inputs, question)
scores = logits(backend, inputs) # uncalibrated (batch, options) scores
```

For a batch, pass a vector of questions, one per row. Yes/no questions
(`NoulQuestion`) use the false then true columns; score questions return an
expected value on a zero-based scale. Unused option columns are excluded
before the softmax, which uses the checkpoint's temperature. Choice confidence
follows Jeff's formula and is not the winning probability.

## NVIDIA GPUs (CUDA)

Install CUDA.jl in your environment (the `tools` environment already has it),
import it and select the device:

```julia
using JeffClient, CUDA
CUDA.allowscalar(false)
CUDA.device!(0)
backend = NativeBackend(checkpoint; device=:cuda)
scores = logits(backend, inputs)
```

The extension runs Float32 Julia CUDA kernels (recurrent Gated DeltaNet,
fused convolution and normalization, batched full attention) and cuBLAS
products. Scratch buffers belong to each model, are reused only after its
CUDA stream completes, and warm forwards allocate no GPU memory; calls sharing
a model are serialized. Returned scores are CPU arrays. The forward selects
the model's device even if the caller has switched to another one. cuDNN is
not required, and CUDA runtime 12.8 and 13.4 have both been verified.

Set `ENV["QDC_CUDA_TRIM_PADDING"] = "1"` to skip leading padding. Interior mask
holes are kept. It is off by default, so the full sequence is computed.

## Apple GPUs (Metal)

On a Mac with Apple Silicon, use the tools environment, which includes
Metal.jl:

```bash
julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=tools examples/metal_inference.jl
```

```julia
import Metal
backend = NativeBackend(checkpoint; device=:metal)
result = decide(backend, inputs, question)
```

The Metal extension uses Metal internal APIs adapted from Laya.jl, so it is
pinned to the verified Metal.jl 1.11.1.

## CPU

CPU execution picks its platform configuration automatically: Accelerate on
Apple Silicon macOS, portable vector math and Julia-parallel projections
elsewhere. The BLAS thread count is set when QwenDecisionCore loads, which
also affects other BLAS users in the process. Start Julia with several
threads (`--threads=8`); a single thread is several times slower for this
hybrid model.

## Limits

Float32 text inference only, with default partial RoPE and bias-free
projections. Batches run one row at a time unless an accelerator provides a
batched path (experimental on Metal). Tokenization, images, generation/KV
cache and training are not implemented.

## Validation

`tools/verify.jl` compares `logits` with an independent PyTorch reference
(see [Development](development.md)). For the pinned checkpoint, 15 cases
(lengths 1–512, batches of 2–3, left padding and interior mask holes) agree
within `1.1e-5` on CUDA, with and without padding trimming.
