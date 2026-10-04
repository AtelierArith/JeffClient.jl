# JeffClient.jl

Julia inference infrastructure for [Jeff](https://github.com/firelex/jeff).
The implementation runs **ONNX models and a native Julia Qwen3.5 model with
prepared input tensors**. Jeff-Qwen3.5-0.8B has been compared with PyTorch on
ONNX CPU, native Julia CPU, and native Julia Metal. Text tokenization and
additional accelerators are planned in [development plan](https://github.com/AtelierArith/JeffClient.jl/blob/main/PLAN.md).

Run `julia --project examples/prepared_inference.jl` for a working CPU demo
using the tiny test graph. It demonstrates the inference boundary and probability
calibration; it does not classify text or use Jeff's trained weights.

The Julia dependency is named `ONNXRunTime` (capital R and T). Python is not
required by this inference API. ONNX Runtime itself is a native library.

## Architecture

The shared infrastructure lives in
[QwenDecisionCore.jl](https://github.com/AtelierArith/QwenDecisionCore.jl), a
Git submodule under `extern/QwenDecisionCore.jl`: the Qwen3.5 / Qwen3.8 backbone
forward pass, the safetensors reader, the Hugging Face checkpoint resolver, the
automatic CPU policy, and the Metal / CUDA / Accelerate / Octavian / SIMD
extensions. JeffClient adds only the Jeff-specific parts: the
`decision_config.json` / `readout.safetensors` bundle, the linear readout and
the ONNX export backend. This mirrors
[KevClient.jl](https://github.com/AtelierArith/KevClient.jl). Clone with
`git clone --recurse-submodules`, or initialise the submodule afterwards with
`git submodule update --init --recursive`.

## Documentation

- [Models and ONNX export](models.md)
- [Inference and accelerators](inference.md)
- [CPU configuration and benchmarking](performance.md)
- [Profiling and measurements](profiling.md)
- [API reference](api.md)
- [Development](development.md)
