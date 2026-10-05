# JeffClient.jl

Native Julia inference for [Jeff](https://github.com/firelex/jeff) decision
models. `NativeBackend` loads a Jeff-Qwen3.5 checkpoint's safetensors and runs
the whole forward pass in Julia on CPU, NVIDIA GPUs (CUDA.jl) or Apple GPUs
(Metal.jl). Inputs are prepared token ids; tokenization and generation are not
part of this package, and Python is used only by the reference tools.

## Architecture

The shared infrastructure lives in
[QwenDecisionCore.jl](https://github.com/AtelierArith/QwenDecisionCore.jl),
fetched from GitHub through the `[sources]` table of `Project.toml`: the
Qwen3.5 / Qwen3.8 backbone forward pass, the safetensors reader, the Hugging
Face checkpoint resolver, the automatic CPU policy and the CUDA / Metal / CPU
acceleration extensions. JeffClient adds only what is Jeff-specific: the
`decision_config.json` / `readout.safetensors` bundle, the linear readout and
the answer calibration. This mirrors
[KevClient.jl](https://github.com/AtelierArith/KevClient.jl).

## Documentation

- [Models](models.md)
- [Inference on CPU, CUDA and Metal](inference.md)
- [Performance](performance.md)
- [API reference](api.md)
- [Development](development.md)
