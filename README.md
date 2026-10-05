# JeffClient.jl

Native Julia inference for [Jeff](https://github.com/firelex/jeff) decision
models, running Qwen3.5 on CPU, NVIDIA GPUs (CUDA.jl) and Apple GPUs
(Metal.jl). Inference accepts prepared token tensors; text tokenization is not
implemented. Python is used only by the reference tools, through PythonCall.jl.

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

## CPU threads

CPU inference chooses its configuration automatically: Accelerate on Apple
Silicon, portable vector math and Julia-parallel projections elsewhere. Start
Julia with several threads (`--threads=8`); with a single Julia thread the
fused task-parallel kernels are disabled and the forward is several times
slower.

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
