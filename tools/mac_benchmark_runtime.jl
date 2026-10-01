include("python_env.jl")
using PythonCall
length(ARGS) == 1 || error("Usage: mac_benchmark_runtime.jl OUTPUT_JSON")
pyexec(
    raw"""
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import torch
versions = {}
for name in ("torch", "transformers", "mlx", "mlx-lm", "numpy", "safetensors"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
report = {"python": sys.version, "python_executable": sys.executable,
          "platform": platform.platform(), "versions": versions,
          "torch_parallel_info": torch.__config__.parallel_info(),
          "thread_probe_note": "Untuned metadata process; use each inference record for actual benchmark thread counts",
          "torch_mps_available": torch.backends.mps.is_available()}
Path(output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
""",
    pydict("output" => abspath(ARGS[1])),
)
