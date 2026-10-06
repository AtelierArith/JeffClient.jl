# Development

Julia 1.13 is the development baseline.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The offline suite runs a small synthetic Qwen fixture (`test/fixtures/native`)
against independent PyTorch reference scores and checks calibration and
configuration errors. It downloads no weights and needs no Python.

`test/cuda.jl` is an optional hardware suite for a functional NVIDIA GPU
(shapes, masks, GC, scratch ownership, concurrency, failure recovery and
device restoration):

```bash
julia --project=tools test/cuda.jl
```

`test/amdgpu.jl` is the corresponding suite for a functional AMD GPU. It needs
an environment whose QwenDecisionCore checkout provides the AMDGPU extension,
which is easiest with a local `Pkg.develop` of `extern/QwenDecisionCore.jl`:

```bash
julia --project=YOUR_ENV test/amdgpu.jl
```

## Tools

The `tools` environment holds CUDA.jl, AMDGPU.jl, Metal.jl and PythonCall:

```bash
julia --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)'
```

| Script | Purpose |
| --- | --- |
| `benchmark.jl` | Warm, synchronized timing on CPU, CUDA, Metal or AMDGPU ([Performance](performance.md)) |
| `verify.jl` | Compare logits on CPU, CUDA, Metal or AMDGPU with a PyTorch reference, twice per case with GC in between |
| `build_reference.jl` | Build that reference for a real checkpoint (PyTorch via PythonCall) |
| `benchmark_pytorch.jl` | Time the original Python implementation on the same input |
| `build_native_fixture.jl`, `build_cuda_reference.jl` | Regenerate the test fixture and its CUDA probes |
| `download_checkpoint.jl` | Fetch a checkpoint into the cache |

For example, to validate the pinned checkpoint on the second GPU:

```bash
julia --project=tools tools/build_reference.jl CHECKPOINT artifacts/reference.json
julia --project=tools tools/verify.jl cuda CHECKPOINT artifacts/reference.json 1
```

The Python tools use the environment of the reference checkout in
`extern/jeff` (`git submodule update --init extern/jeff`, then create its
virtual environment as its README describes); `JULIA_PYTHONCALL_EXE`
overrides the interpreter.

## Continuous integration

The [CI workflow](https://github.com/AtelierArith/JeffClient.jl/blob/main/.github/workflows/CI.yml)
runs the CPU test suite with Julia 1.13 on Ubuntu x64 for pushes to `main`,
tags, pull requests and manual dispatch. GPU suites are run by hand.

## Build the documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=docs docs/make.jl
```

HTML is written to `docs/build/` (ignored by Git). The documentation workflow
builds pull requests and publishes `main` to the `gh-pages` branch.
