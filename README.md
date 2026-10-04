# JeffClient.jl

Native Julia inference for [Jeff](https://github.com/firelex/jeff), running
Qwen3.5 on CPU and Apple GPU through Metal.jl. Inference accepts prepared token
tensors; text tokenization is not implemented. Python is needed only for export
and reference tools, through PythonCall.jl.

## Architecture

The shared infrastructure lives in
[QwenDecisionCore.jl](https://github.com/AtelierArith/QwenDecisionCore.jl), a
Git submodule under `extern/QwenDecisionCore.jl`: the Qwen3.5 / Qwen3.8 backbone
forward pass, the safetensors reader, the Hugging Face checkpoint resolver, the
automatic CPU policy and the Metal / CUDA / Accelerate / Octavian / SIMD
extensions. JeffClient is a thin client that adds only what is Jeff-specific:
the `decision_config.json` / `readout.safetensors` bundle, the linear readout and
the ONNX export backend. This mirrors
[KevClient.jl](https://github.com/AtelierArith/KevClient.jl).

Neither package is registered on the General registry, so clone with the
submodule:

```bash
git clone --recurse-submodules https://github.com/AtelierArith/JeffClient.jl.git
# in an existing checkout:
git submodule update --init --recursive
```

## Quick start: native Julia on CPU

Run [examples/native_inference.jl](examples/native_inference.jl) with Jeff's
actual 0.8B weights. First install Julia through
[juliaup](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh   # then reopen the terminal
juliaup add 1.13 && juliaup default 1.13
```

From the repository root (internet access is needed for dependencies and the
first model download):

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project examples/native_inference.jl
```

This downloads the pinned checkpoint if it is not cached (about 1.7 GB) and uses
a bundled, pre-tokenized parcel inquiry. The native backend loads safetensors
weights and computes the model in Julia on CPU, using LinearAlgebra/BLAS for
matrix products. No ONNX model, export, or Python setup is required. Actual CPU
demo output:

```text
Device: cpu
Input: The parcel arrived crushed and the customer wants a replacement.
Question: Which team should handle this?
Choice: delivery
  refund: 0.003434
  delivery: 0.996566
Confidence: 0.993133
```

Pass a local checkpoint directory as the first argument to reuse your weights:

```bash
julia --project examples/native_inference.jl CHECKPOINT_DIRECTORY
```

The fixed prompt runs entirely in Julia; editing its text requires regenerating
the tokens. See the [inference page](docs/src/inference.md) for details and
Metal.

### Apple GPU (Metal)

On a Mac with an Apple GPU, set up the tools environment, which includes
Metal.jl, then run [examples/metal_inference.jl](examples/metal_inference.jl):

```bash
julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=tools examples/metal_inference.jl
# Or reuse a local checkpoint:
julia --project=tools examples/metal_inference.jl CHECKPOINT_DIRECTORY
```

This executes the same real-checkpoint demo on an Apple GPU and prints
`Device: metal`. It shares the CPU demo's model cache; Python and ONNX export are
not required. The first run compiles GPU kernels, so startup is longer.

### Optimized CPU execution

CPU inference automatically chooses its platform configuration; no tuning
environment variables are needed. Apple Silicon macOS uses Accelerate; Intel and
other platforms use portable vector math and Julia-parallel projections. The
BLAS thread count is set automatically when the core loads (this affects other
BLAS users in the same process). Start Julia with multiple workers to enable
parallel kernels:

```bash
julia --threads=8 --project=tools examples/native_inference.jl
```

This runs on CPU without Metal or MKL. See the performance page for conditions.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The offline suite checks the ONNX backend (answers, calibration, export metadata)
and the native backend against a committed independent PyTorch reference. The
optional `test/cuda.jl` hardware suite requires a functional NVIDIA GPU.

## Documentation

Detailed documentation is maintained with Documenter.jl in [`docs/`](docs):

- [Overview](docs/src/index.md)
- [Model downloads and caching](docs/src/models.md)
- [Inference, CUDA and Metal](docs/src/inference.md)
- [Performance comparisons with Python](docs/src/performance.md)
- [Profiling, allocation findings and Metal tuning](docs/src/profiling.md)
- [API reference](docs/src/api.md)
- [Development and documentation builds](docs/src/development.md)

See [measured inference speed](docs/src/profiling.md) for CPU comparisons,
conditions and reproducible commands. On Apple Silicon macOS, reproduce the
matched CPU/GPU comparisons with `./tools/mac-M-series.sh` (the one-thread CPU
comparison and the PyTorch 8 / Julia 8 with automatic Accelerate comparison run
by default; `--help` lists the options). Joint GPU batching remains experimental.

The [development plan](PLAN.md), [measurement notes](memories/MEMORY.md) and
[open issues](https://github.com/AtelierArith/JeffClient.jl/issues) track
remaining work.
