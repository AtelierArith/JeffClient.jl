# JeffClient.jl

Native Julia inference for [Jeff](https://github.com/firelex/jeff) decision
models, running Qwen3.5 on CPU, NVIDIA GPUs (CUDA.jl) and Apple GPUs
(Metal.jl). Inference accepts prepared token tensors; text tokenization is not
implemented. Python is used only by the reference tools, through PythonCall.jl.

## Minimal example

After the [setup](#setup), run this from the repository root with
`julia --threads=8 --project`. The first run downloads the pinned checkpoint
(about 1.7 GB). It classifies a bundled, pre-tokenized parcel inquiry on CPU:

```julia
using JeffClient
import JSON

sample = JSON.parsefile("examples/data/parcel.json")
checkpoint = resolve_checkpoint(sample["model"]; revision = sample["revision"])
backend = NativeBackend(checkpoint)  # device = :metal needs `import Metal`

criteria = sample["question"]["criteria"]
question = ChoiceQuestion(
    ["refund" => criteria["refund"], "delivery" => criteria["delivery"]];
    instructions = sample["question"]["instructions"],
)
# Prepared token tensors; text tokenization is not implemented.
inputs = Dict(
    name => reduce(vcat, [permutedims(Int64.(row)) for row in rows]) for
    (name, rows) in sample["inputs"]
)

result = decide(backend, inputs, question)
result.choice        # "delivery"
result.probabilities # refund ≈ 0.0034, delivery ≈ 0.9966
result.confidence    # ≈ 0.9931
```

For Metal, use `--project=tools`, `import Metal` and
`NativeBackend(checkpoint; device = :metal)`; see
[Apple GPU (Metal)](#apple-gpu-metal).

## Architecture

The shared infrastructure lives in
[QwenDecisionCore.jl](https://github.com/AtelierArith/QwenDecisionCore.jl): the
Qwen3.5 / Qwen3.8 backbone forward pass, the safetensors reader, the Hugging
Face checkpoint resolver, the automatic CPU policy and the CUDA / Metal / CPU
acceleration extensions. JeffClient is a thin client that adds only what is
Jeff-specific: the `decision_config.json` / `readout.safetensors` bundle, the
linear readout and the answer calibration. This mirrors
[KevClient.jl](https://github.com/AtelierArith/KevClient.jl).

QwenDecisionCore is not registered; `Project.toml` points at its GitHub
repository through `[sources]`, so `Pkg.instantiate()` clones it.

## Setup

Requirements: Julia 1.13 and Git. CUDA needs an NVIDIA GPU; Metal needs a Mac
with Apple Silicon.

```bash
curl -fsSL https://install.julialang.org | sh   # juliaup; then reopen the terminal
juliaup add 1.13 && juliaup default 1.13

git clone https://github.com/AtelierArith/JeffClient.jl.git
cd JeffClient.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
# GPU examples, benchmarks and validation use the tools environment:
julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
```

If instantiation fails with `empty intersection between QwenDecisionCore`,
delete the stale (git-ignored) `Manifest.toml` files and instantiate again.

## Quick start

```bash
julia --threads=8 --project examples/native_inference.jl
```

The first run downloads the pinned Jeff-Qwen3.5-0.8B checkpoint (about 1.7 GB)
and classifies a bundled, pre-tokenized parcel inquiry on the CPU:

```text
Device: cpu
Input: The parcel arrived crushed and the customer wants a replacement.
Question: Which team should handle this?
Choice: delivery
  refund: 0.003434
  delivery: 0.996566
Confidence: 0.993133
```

Pass a checkpoint directory as the first argument to reuse local weights. The
same demo runs on an NVIDIA GPU with
`julia --project=tools examples/native_inference.jl CHECKPOINT cuda`, and on a
Mac's GPU with `julia --project=tools examples/metal_inference.jl`.

## CUDA

```julia
using JeffClient, CUDA
CUDA.allowscalar(false)
backend = NativeBackend("mstrasser/Jeff-Qwen3.5-0.8B"; device=:cuda)
scores = logits(backend, inputs)
```

On an RTX 3060 (Float32, batch 1, 256 tokens of which 101 active, CUDA
runtime 13.4) a warm forward takes a median **70.6 ms** for the full sequence
and **34.6 ms** with `QDC_CUDA_TRIM_PADDING=1`, with no GPU allocation per
forward. Reproduce with:

```bash
julia --threads=8 --project=tools tools/benchmark.jl cuda
```

See [Performance](docs/src/performance.md) for the conditions.

## Apple GPU (Metal)

On a Mac with an Apple GPU, set up the tools environment, which includes
Metal.jl, then run [examples/metal_inference.jl](examples/metal_inference.jl):

```bash
julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=tools examples/metal_inference.jl
# Or reuse a local checkpoint:
julia --project=tools examples/metal_inference.jl CHECKPOINT_DIRECTORY
```

This executes the same real-checkpoint demo on an Apple GPU and prints
`Device: metal`. It shares the CPU demo's model cache; ONNX export is not
required. The first run compiles GPU kernels, so startup is longer (about
30 s on an Apple M4, Julia 1.13.1, Metal.jl 1.11.1).

To check the Metal numerics against an independent PyTorch reference (this needs
the `extern/jeff` submodule and the Python tools environment):

```bash
julia --project=tools tools/build_reference.jl CHECKPOINT_DIRECTORY reference.json
julia --project=tools tools/verify.jl metal CHECKPOINT_DIRECTORY reference.json
```

It runs padded and mixed-length cases twice with a GC between cases and fails if
any logit differs beyond tolerance.

## CPU threads

CPU inference selects its platform configuration automatically. On Apple
silicon the Accelerate fast path is opt-in: add `AppleAccelerate` to your
environment and `using AppleAccelerate` before the forward (`QwenDecisionCore`'s
extension forwards BLAS to Accelerate in either load order). The `tools`
environment already includes it, and `tools/benchmark.jl` exposes
`--accelerate`; without it Julia uses the default BLAS (OpenBLAS).

> **Apple silicon note.** Accelerate used to be enabled automatically. Since
> `AppleAccelerate` became an opt-in weak dependency, a plain `using JeffClient`
> runs the slower default BLAS. If you want the previous Accelerate speed, add
> `AppleAccelerate` to your project and `using AppleAccelerate`, or run with
> `--project=tools`.

The portable vector math and Julia-parallel projections run on every platform,
including Linux. Start Julia with several threads (`--threads=8`); with a single
Julia thread the fused task-parallel kernels are disabled and the forward is
several times slower.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The offline suite checks the native backend against a committed independent
PyTorch reference, plus calibration and configuration errors. The optional
`test/cuda.jl` hardware suite requires a functional NVIDIA GPU.

## Documentation

Documenter.jl sources live in [`docs/`](docs):

- [Overview](docs/src/index.md)
- [Models](docs/src/models.md)
- [Inference on CPU, CUDA and Metal](docs/src/inference.md)
- [Performance](docs/src/performance.md)
- [API reference](docs/src/api.md)
- [Development, tools and documentation builds](docs/src/development.md)

Implementation notes and measurement history are kept in
[memories/MEMORY.md](memories/MEMORY.md).
