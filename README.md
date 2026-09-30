# JeffClient.jl

Julia inference for [Jeff](https://github.com/firelex/jeff), using ONNX Runtime
or a native Julia Qwen3.5 implementation on CPU and Apple GPU through Metal.jl.
Inference accepts prepared token tensors; text tokenization is not implemented.
Python is needed only for export and reference tools, through PythonCall.jl.

## Quick start

With Julia 1.13, instantiate the project and run the tiny CPU demo:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project examples/prepared_inference.jl
```

The demo uses a test graph, not Jeff's trained weights. For a real checkpoint:

```julia
using JeffClient

checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990")
backend = NativeBackend(checkpoint) # CPU
# import Metal; NativeBackend(checkpoint; device=:metal) for Apple GPU
# logits(backend, inputs) accepts prepared input_ids and attention_mask.
```

## Documentation

Detailed documentation is maintained with Documenter.jl in [`docs/`](docs):

- [Overview](docs/src/index.md)
- [Model downloads, caching and ONNX export](docs/src/models.md)
- [Inference, CUDA and Metal](docs/src/inference.md)
- [Performance comparisons with Python](docs/src/performance.md)
- [Profiling, allocation findings and Metal tuning](docs/src/profiling.md)
- [API reference](docs/src/api.md)
- [Development and documentation builds](docs/src/development.md)

On the measured Apple M4 Float32 inputs, Metal inference is about 1.8× faster
than Python MPS by default, and 2.2–4.1× with padding-related optimizations.
These results depend on inputs and settings; see the performance page for
conditions and limitations. Joint GPU batching remains experimental.

The [development plan](PLAN.md), [measurement notes](memories/MEMORY.md), and
[open issues](https://github.com/AtelierArith/JeffClient.jl/issues) track remaining work.
