include("python_env.jl")
using PythonCall

context = pydict("destination" => joinpath(@__DIR__, "..", "test", "fixtures", "native"))
pyexec(
    raw"""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

torch.manual_seed(731)
torch.set_num_threads(1)
config = Qwen3_5TextConfig(
    vocab_size=64, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=4,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=1, linear_num_value_heads=2,
    linear_key_head_dim=2, linear_value_head_dim=2, linear_conv_kernel_dim=4,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 0]},
    attention_bias=False, hidden_act="silu", max_position_embeddings=128,
)
model = Qwen3_5TextModel(config).eval()
with torch.no_grad():
    for name, value in model.named_parameters():
        if value.ndim > 1:
            value.normal_(0.0, 0.15)
readout = torch.randn(3, 8) * 0.25
path = Path(destination)
path.mkdir(parents=True, exist_ok=True)
save_file({"language_model." + k: v for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
save_file({"weight": readout}, str(path / "readout.safetensors"))
(path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": config.to_dict()}))
(path / "decision_config.json").write_text(json.dumps({"format_version": 1, "temperature": 1.25, "max_options": 3}))
# This prepared-token fixture deliberately does not include a real tokenizer.
for name in ("tokenizer.json", "tokenizer_config.json"):
    (path / name).write_text("{}")
cases = []
with torch.inference_mode():
    for length in (1, 3, 63, 64, 65):
        ids = torch.randint(0, 64, (2, length))
        mask = torch.ones_like(ids)
        mask[1, :length // 2] = 0
        hidden = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
        cases.append({"inputs": {"input_ids": ids.tolist(), "attention_mask": mask.tolist()},
                      "logits": (hidden @ readout.T).tolist()})
(path / "reference.json").write_text(json.dumps(cases))
""",
    context,
)
