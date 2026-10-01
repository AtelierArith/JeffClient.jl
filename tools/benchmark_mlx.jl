include("python_env.jl")
using PythonCall

length(ARGS) in 4:6 || error(
    "Usage: julia --project=tools tools/benchmark_mlx.jl CHECKPOINT cpu|gpu REFERENCE_JSON OUTPUT_JSON [SAMPLES] [reference-norm]",
)
context = pydict(
    "checkpoint" => abspath(ARGS[1]),
    "device_name" => ARGS[2],
    "reference_path" => abspath(ARGS[3]),
    "output" => abspath(ARGS[4]),
    "samples" => length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 30,
    "reference_norm" => length(ARGS) == 6 && ARGS[6] == "reference-norm",
    "jeff_source" => abspath(joinpath(@__DIR__, "..", "extern", "jeff", "src")),
)
pyexec(
    raw"""
import gc
import importlib.metadata
import json
import math
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, jeff_source)
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx.utils import tree_flatten
from types import SimpleNamespace
import inspect
import textwrap
import mlx_lm.models.qwen3_5 as qwen

if device_name not in ("cpu", "gpu") or samples < 3:
    raise ValueError("Choose cpu or gpu, with at least three samples")
mx.set_default_device(mx.cpu if device_name == "cpu" else mx.gpu)
if reference_norm:
    # Transformers adds epsilon to SUM(q*q); MLX adds it to MEAN(q*q).
    # Divide epsilon by head width to match the reference mathematics. Keep
    # the original MLX implementation/kernels and apply this only in-process.
    method_source = textwrap.dedent(inspect.getsource(qwen.GatedDeltaNet.__call__))
    old = "mx.fast.rms_norm(q, None, 1e-6)"
    old_k = "mx.fast.rms_norm(k, None, 1e-6)"
    if method_source.count(old) != 1 or method_source.count(old_k) != 1:
        raise RuntimeError("MLX normalization adapter no longer matches upstream")
    method_source = method_source.replace(old, "mx.fast.rms_norm(q, None, 1e-6 / q.shape[-1])").replace(old_k, "mx.fast.rms_norm(k, None, 1e-6 / k.shape[-1])")
    namespace = dict(vars(qwen))
    exec(compile(method_source, "<mlx-reference-normalization>", "exec"), namespace)
    qwen.GatedDeltaNet.__call__ = namespace["__call__"]
report = json.loads(Path(reference_path).read_text())
reference = (report["cases"] if isinstance(report, dict) else report)[0]
start = time.perf_counter()
directory = Path(checkpoint)
config = json.loads((directory / "config.json").read_text())
mlx_model = Model(ModelArgs.from_dict(config))
raw = mx.load(str(directory / "model.safetensors"))
# Cast BEFORE sanitize: normalization shifts must be added in Float32, just as
# in the CPU loaders, rather than rounded in BF16 and subsequently widened.
renamed = {"model." + name: value.astype(mx.float32)
           for name, value in raw.items() if name.startswith("language_model.")}
mlx_model.load_weights(list(mlx_model.sanitize(renamed).items()), strict=True)
mlx_model.eval()
readout = mx.load(str(directory / "readout.safetensors"))["weight"].astype(mx.float32)
backend = SimpleNamespace(model=mlx_model, readout=readout)
mx.eval(mlx_model.parameters(), readout)
del raw, renamed
model = backend.model.language_model.model
load_seconds = time.perf_counter() - start
ids = mx.array(reference["inputs"]["input_ids"], dtype=mx.int32)
padding = mx.array(reference["inputs"]["attention_mask"], dtype=mx.bool_)
length = ids.shape[1]
# Keep all positions, including padding. Pass the same masks into the original
# MLX layers without a persistent KV/recurrent cache. No prompt/tokenization.
causal = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
attention_mask = causal[None, None, :, :] & padding[:, None, None, :]
mx.eval(ids, padding, attention_mask)
host_ids = np.array(reference["inputs"]["input_ids"], dtype=np.int32)
host_padding = np.array(reference["inputs"]["attention_mask"], dtype=bool)
parameter_dtypes = sorted({str(x.dtype) for _, x in tree_flatten(backend.model.parameters())})
if parameter_dtypes != ["mlx.core.float32"]:
    raise RuntimeError(f"Expected all Float32 weights, got {parameter_dtypes}")

def forward():
    # Start from CPU prepared inputs on every call, like the Julia GPU API.
    ids = mx.array(host_ids)
    padding = mx.array(host_padding)
    causal = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
    attention_mask = causal[None, None, :, :] & padding[:, None, None, :]
    hidden = model.embed_tokens(ids)
    for layer in model.layers:
        hidden = layer(hidden, mask=padding if layer.is_linear else attention_mask, cache=None)
    hidden = model.norm(hidden)
    scores = hidden[:, -1, :] @ backend.readout.T
    mx.eval(scores)
    mx.synchronize()
    # Include materialization of CPU scores, as in the other implementations.
    return np.array(scores)

start = time.perf_counter()
actual = forward()
first_seconds = time.perf_counter() - start
expected = np.array(reference["logits"], dtype=np.float32)
max_error = float(np.max(np.abs(actual - expected)))
try:
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
except AssertionError:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"backend": "mlx-" + device_name,
        "reference_norm": reference_norm, "validation_passed": False,
        "max_logit_error": max_error, "sequence_length": length,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm")}, indent=2) + "\n")
    raise
forward()
gc.collect()
times = []
for _ in range(samples):
    start = time.perf_counter()
    forward()
    times.append((time.perf_counter() - start) * 1000)
result = {
    "validation_passed": True, "reference_norm": reference_norm,
    "backend": "mlx-" + device_name,
    "mlx_version": importlib.metadata.version("mlx"),
    "mlx_lm_version": importlib.metadata.version("mlx-lm"),
    "device": str(mx.default_device()),
    "weight_dtypes": parameter_dtypes,
    "batch_size": ids.shape[0], "sequence_length": length,
    "computed_tokens": length, "active_tokens": np.array(padding).sum(-1).tolist(),
    "samples": len(times), "times_ms": times,
    "median_ms": statistics.median(times), "minimum_ms": min(times),
    "maximum_ms": max(times), "p95_ms": sorted(times)[math.ceil(.95*len(times))-1],
    "model_load_seconds": load_seconds, "first_forward_seconds": first_seconds,
    "max_logit_error": max_error,
    "cpu_input_upload_included": True,
    "mask_adapter": "original layers with explicit padding/causal masks; no persistent cache",
}
print(json.dumps(result, indent=2), flush=True)
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(result, indent=2) + "\n")
""",
    context,
)
