include("python_env.jl")
using PythonCall

length(ARGS) in 4:6 || error(
    "Usage: julia --project=tools tools/benchmark_original.jl CHECKPOINT cpu|mps|mps-f32 REFERENCE_JSON SAMPLES [OUTPUT_JSON] [CASE_INDEX]",
)
context = pydict(
    "checkpoint" => abspath(ARGS[1]),
    "mode" => ARGS[2],
    "reference_path" => abspath(ARGS[3]),
    "samples" => parse(Int, ARGS[4]),
    "output" => length(ARGS) >= 5 ? abspath(ARGS[5]) : nothing,
    "case_index" => length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1,
    "jeff_source" => joinpath(@__DIR__, "..", "extern", "jeff", "src"),
)
pyexec(
    raw"""
import json
import math
import statistics
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, jeff_source)
from jeff.models import load_decision_model
from jeff.model import PreparedBatch

if mode not in ("cpu", "mps", "mps-f32") or samples < 3:
    raise ValueError("Choose cpu, mps, or mps-f32, and at least three samples")
device = "cpu" if mode == "cpu" else "mps"
if device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")
report = json.loads(Path(reference_path).read_text())
cases = report["cases"] if isinstance(report, dict) else report
if not 1 <= case_index <= len(cases):
    raise ValueError("CASE_INDEX is outside the reference cases")
reference = cases[case_index - 1]
if "question" in reference:
    question = reference["question"]
    count = 2 if question["type"] == "noul" else len(question["criteria"])
else:
    # Expanded numerical probes have no question; preserve the checkpoint's
    # trained option limit while exercising the original full forward.
    count = json.loads((Path(checkpoint) / "decision_config.json").read_text())["max_options"]
start = time.perf_counter()
load_device = "cpu" if mode == "mps-f32" else device
model = load_decision_model(checkpoint=checkpoint, device=load_device, cpu_threads=8).eval()
if mode == "mps-f32":
    # Load float32 directly, so the readout is never rounded through bfloat16.
    model.to("mps")
load_seconds = time.perf_counter() - start
inputs = {k: torch.tensor(v, dtype=torch.int64, device=device)
          for k, v in reference["inputs"].items()}
batch = PreparedBatch(inputs, (count,) * inputs["input_ids"].shape[0],
                      int(inputs["attention_mask"].sum().item()))

def forward():
    # Call the original Jeff forward, including its multimodal backbone wrapper,
    # trained readout, and option mask. Transfer scores back like the Julia API.
    result = model(batch).cpu()
    if device == "mps":
        torch.mps.synchronize()
    return result

with torch.inference_mode():
    start = time.perf_counter()
    actual = forward()
    first_seconds = time.perf_counter() - start
    expected = torch.tensor(reference["logits"], dtype=torch.float32)
    max_error = float((actual[:, :count] - expected[:, :count]).abs().max())
    if not torch.isfinite(actual).all():
        raise RuntimeError("Nonfinite original Jeff logits")
    if mode != "mps":
        torch.testing.assert_close(actual[:, :count], expected[:, :count], atol=2e-4, rtol=2e-4)
    forward()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        forward()
        times.append((time.perf_counter() - start) * 1000)
result = {"backend": "original-python-" + mode, "torch_version": torch.__version__,
    "case_index": case_index,
    "weight_dtype": str(next(model.parameters()).dtype), "cpu_threads": torch.get_num_threads(),
    "batch_size": int(inputs["input_ids"].shape[0]),
    "sequence_length": int(inputs["input_ids"].shape[1]),
    "active_tokens": inputs["attention_mask"].sum(-1).cpu().tolist(), "samples": len(times),
    "model_load_seconds": load_seconds, "first_forward_seconds": first_seconds,
    "median_ms": statistics.median(times), "minimum_ms": min(times), "maximum_ms": max(times),
    "p95_ms": sorted(times)[math.ceil(.95 * len(times)) - 1], "max_active_logit_error": max_error}
print(json.dumps(result, indent=2), flush=True)
if output is not None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
""",
    context,
)
