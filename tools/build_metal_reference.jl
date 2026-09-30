include("python_env.jl")
using PythonCall

length(ARGS) == 2 || error(
    "Usage: julia --project=tools tools/build_metal_reference.jl CHECKPOINT OUTPUT_JSON",
)
context = pydict(
    "checkpoint" => abspath(ARGS[1]),
    "destination" => abspath(ARGS[2]),
    "jeff_source" => joinpath(@__DIR__, "..", "extern", "jeff"),
)
pyexec(
    raw"""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, jeff_source)
from jeff.models import load_decision_model

torch.set_num_threads(8)
torch.manual_seed(731)
model = load_decision_model(checkpoint=checkpoint, device="cpu").eval()
rows = [
    {"state": "The parcel arrived crushed and the customer wants a replacement.",
     "question": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"refund": "Refunds and payments", "delivery": "Damaged parcels"}}},
    {"state": "荷物が破損して届きました。交換をお願いします。",
     "question": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"refund": "Refunds and payments", "delivery": "Damaged parcels"}}},
    {"state": "Everything was perfect. Thanks!",
     "question": {"type": "noul", "instructions": "Is the customer angry?"}},
]
prepared = model.prepare(rows)
cases = []
with torch.inference_mode():
    ids, mask = prepared.inputs["input_ids"], prepared.inputs["attention_mask"]
    expected = model.backbone.language_model(input_ids=ids, attention_mask=mask,
        use_cache=False).last_hidden_state[:, -1]
    scores = model.readout(expected).float()
    original = model(prepared)
    for i, count in enumerate(prepared.counts):
        torch.testing.assert_close(scores[i, :count], original[i, :count], atol=2e-4, rtol=2e-4)
    cases.append({"name": "mixed English/Japanese prompts", "inputs": {
        "input_ids": ids.tolist(), "attention_mask": mask.tolist()}, "logits": scores.tolist()})
    # Exercise every side of the 64-token DeltaNet chunk boundary with real
    # checkpoint weights. Random tokens probe numerical behavior, not accuracy
    # of language classification; the first row is unpadded and the second padded.
    for length in (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512):
        ids = torch.randint(0, model.backbone.language_model.config.vocab_size, (2, length))
        mask = torch.ones_like(ids)
        mask[1, :length // 2] = 0
        hidden = model.backbone.language_model(input_ids=ids, attention_mask=mask,
            use_cache=False).last_hidden_state[:, -1]
        scores = model.readout(hidden).float()
        cases.append({"name": f"length {length}, batch 2", "inputs": {
            "input_ids": ids.tolist(), "attention_mask": mask.tolist()}, "logits": scores.tolist()})
        print(f"Recorded length {length}", flush=True)
    # Zero prefixes, interior holes, and a single final active token must remain
    # equivalent when a backend skips only leading padding.
    for length in (9, 65, 129):
        ids = torch.randint(0, model.backbone.language_model.config.vocab_size, (3, length))
        mask = torch.ones_like(ids)
        prefix = length // 3
        mask[0, :prefix] = 0
        mask[0, prefix + 1:-1:3] = 0
        mask[1, :-1] = 0
        mask[2, 1:-1:2] = 0
        hidden = model.backbone.language_model(input_ids=ids, attention_mask=mask,
            use_cache=False).last_hidden_state[:, -1]
        scores = model.readout(hidden).float()
        cases.append({"name": f"interior masks length {length}, batch 3", "inputs": {
            "input_ids": ids.tolist(), "attention_mask": mask.tolist()}, "logits": scores.tolist()})
        print(f"Recorded interior masks length {length}", flush=True)
report = {"format_version": 1, "checkpoint": str(Path(checkpoint).resolve()),
    "torch_version": torch.__version__, "seed": 731, "cases": cases}
path = Path(destination)
if path.exists():
    raise FileExistsError(f"Choose a new reference output: {path}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(f"Saved {len(cases)} cases to {path}", flush=True)
""",
    context,
)
