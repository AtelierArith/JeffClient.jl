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

The first demo uses a test graph. To run **Jeff's actual 0.8B weights** on CPU:

```bash
julia --project examples/native_inference.jl
```

This downloads the pinned checkpoint if it is not cached (about 1.7 GB) and uses
a bundled, pre-tokenized parcel inquiry. Actual CPU demo output:

```text
Input: The parcel arrived crushed and the customer wants a replacement.
Question: Which team should handle this?
Choice: delivery
  refund: 0.003434
  delivery: 0.996566
Confidence: 0.993133
```

Pass a local checkpoint directory as the first argument to reuse your weights.
The fixed prompt runs entirely in Julia; editing its text requires regenerating
the tokens. See [inference examples](docs/src/inference.md) for details and Metal.

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
