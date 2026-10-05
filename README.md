# JeffClient.jl

Native Julia inference for [Jeff](https://github.com/firelex/jeff), running
Qwen3.5 on CPU, Apple GPU (Metal.jl) and NVIDIA GPU (CUDA). Inference accepts prepared token
tensors; text tokenization is not implemented. Python is needed only for export
and reference tools, through PythonCall.jl.

## Architecture

The shared infrastructure lives in
[QwenDecisionCore.jl](https://github.com/AtelierArith/QwenDecisionCore.jl), a
separate package: the Qwen3.5 / Qwen3.8 backbone forward pass, the safetensors reader, the Hugging Face checkpoint resolver, the
automatic CPU policy and the Metal / CUDA / Accelerate / Octavian / SIMD
extensions. JeffClient is a thin client that adds only what is Jeff-specific:
the `decision_config.json` / `readout.safetensors` bundle, the linear readout and
the ONNX export backend. This mirrors
[KevClient.jl](https://github.com/AtelierArith/KevClient.jl).

It is not registered on the General registry. `Project.toml` points at
`https://github.com/AtelierArith/QwenDecisionCore.jl.git` through `[sources]`, so
`Pkg.instantiate()` clones it automatically and no submodule is involved.

## Setup guide

Requirements: Julia 1.13 and Git. The Metal path additionally needs a Mac with
Apple Silicon (arm64 macOS).

1. Install Julia through [juliaup](https://github.com/JuliaLang/juliaup):

   ```bash
   curl -fsSL https://install.julialang.org | sh   # then reopen the terminal
   juliaup add 1.13 && juliaup default 1.13
   ```

2. Clone the repository. `extern/jeff` is the only submodule; it holds the
   Python reference used by the tools and is not needed for inference.

   ```bash
   git clone https://github.com/AtelierArith/JeffClient.jl.git
   cd JeffClient.jl
   # optional, for the Python reference tools and reference tests:
   git submodule update --init extern/jeff
   ```

3. Install the dependencies. This fetches QwenDecisionCore.jl from GitHub, so
   internet access is required.

   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

   For Metal, benchmarks and profiling, instantiate the tools workspace instead:

   ```bash
   julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
   ```

4. Run the demo. The first run downloads the pinned checkpoint (about 1.7 GB):

   ```bash
   julia --threads=8 --project examples/native_inference.jl
   ```

5. Optionally verify the installation with `Pkg.test()` (see [Tests](#tests)).

If instantiation fails with `empty intersection between QwenDecisionCore` or
cannot find the package, delete the stale `Manifest.toml` files (they are
git-ignored) and instantiate again.

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
not required. The first run compiles GPU kernels, so startup is longer (about
30 s on an Apple M4, Julia 1.13.1, Metal.jl 1.11.1).

To check the Metal numerics against an independent PyTorch reference (this needs
the `extern/jeff` submodule and the Python tools environment):

```bash
julia --project=tools tools/build_metal_reference.jl CHECKPOINT_DIRECTORY reference.json
julia --project=tools tools/verify_metal.jl CHECKPOINT_DIRECTORY reference.json
```

It runs padded and mixed-length cases twice with a GC between cases and fails if
any logit differs beyond tolerance.

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

Multiple Julia threads are worth it: they enable the fused task-parallel kernels
and, when `Threads.nthreads() == 1`, `initialize_cpu!` sets BLAS to the
physical-core count instead, which is much slower for this hybrid model.
Measured on an Intel i9-9900K (8 physical cores), a 0.8B forward took about
128 / 120 / 199 ms at 8 / 16 / 64 tokens with `--threads=8`, versus about
842 / 893 / 716 ms with the default single-threaded Julia.

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
