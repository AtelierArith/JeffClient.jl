# Regenerate CUDA's mask/shape probes from the existing independent tiny fixture.
ENV["USE_HUB_KERNELS"] = "NO"
include("python_env.jl")
using PythonCall

context = pydict("destination"=>joinpath(@__DIR__, "..", "test", "fixtures", "native"))
pyexec(
    raw"""
import inspect, json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
    setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
torch.set_num_threads(1)
torch.manual_seed(932)
path = Path(destination)
cfg = Qwen3_5TextConfig(**json.loads((path/'config.json').read_text())['text_config'])
model = Qwen3_5TextModel(cfg).eval()
model.load_state_dict({k.removeprefix('language_model.'): v
                      for k, v in load_file(str(path/'model.safetensors')).items()})
readout = load_file(str(path/'readout.safetensors'))['weight']
cases = []
with torch.inference_mode():
    for length in (1, 9, 63, 64, 65, 129, 256):
        ids = torch.randint(0, 64, (3, length))
        for pattern in ('prefix', 'holes', 'last-only'):
            mask = torch.ones_like(ids)
            if pattern == 'prefix':
                mask[1, :length//2] = 0
            elif pattern == 'holes':
                mask[1, 1:-1:2] = 0
                mask[2, :length//3] = 0
            else:
                mask[1, :-1] = 0
            hidden = model(input_ids=ids, attention_mask=mask,
                           use_cache=False).last_hidden_state[:, -1]
            cases.append({'name': f'{length}-{pattern}',
                          'inputs': {'input_ids': ids.tolist(), 'attention_mask': mask.tolist()},
                          'logits': (hidden @ readout.T).tolist()})
(path/'cuda_reference.json').write_text(json.dumps(cases))
print('Saved', len(cases), 'independent reference cases')
""",
    context,
)
