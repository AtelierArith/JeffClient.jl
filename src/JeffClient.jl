"""
    JeffClient

Unofficial Julia inference for [Jeff](https://github.com/firelex/jeff) decision
models. The heavy lifting — the Qwen3.5 / Qwen3.8 backbone, safetensors
loading, Hugging Face checkpoint resolution, the CPU policy and the accelerator
extensions — lives in
[QwenDecisionCore](https://github.com/AtelierArith/QwenDecisionCore.jl).
JeffClient adds what is Jeff-specific: the `decision_config.json` /
`readout.safetensors` bundle and the linear readout.

Inference runs on prepared token ids; no tokenization or generation is
performed.
"""
module JeffClient

import JSON
using LinearAlgebra
using QwenDecisionCore
# Diagnostic CPU policy is owned by QwenDecisionCore; re-export it so existing
# scripts keep working.
using QwenDecisionCore: cpu_settings, with_cpu_settings

export ChoiceQuestion, NoulQuestion, ScoreQuestion, decide
export resolve_checkpoint
export NativeBackend, logits
export QwenBackbone, backbone_hidden
export cpu_settings, with_cpu_settings

include("native.jl")

end # module JeffClient
