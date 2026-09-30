# Models and ONNX export

## Model downloads and cache

```julia
using JeffClient

checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990")
# Reuse locally, with network access disabled:
checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990", offline=true)
```

This downloads the **source checkpoint**, not an ONNX model. The loader checks
the existing Hugging Face cache, then JeffClient's Scratch.jl cache under your
Julia depot. Only model/config/tokenizer/license files are fetched; videos and
assets are skipped. Supply `cache_dir="..."` to choose another cache root.
`HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`, and `HF_HUB_OFFLINE` are
supported. Local checkpoint directories can also be passed directly.

## Export and validate Jeff

The one-time export uses the Python environment for `extern/jeff` plus
`tools/requirements.txt`, accessed from Julia through PythonCall.jl. Download the source weights using Julia above, then
pass the returned checkpoint directory to the exporter:

```bash
uv pip install --python extern/jeff/.venv/bin/python -r tools/requirements.txt
julia --project=tools tools/export_onnx.jl CHECKPOINT_DIRECTORY artifacts/jeff-0.8b-onnx \
    --sequence-length 256 --source-revision 0f212b3e72acb4dde3f7da61e925d6ab7f819990
julia --project examples/verify_export.jl artifacts/jeff-0.8b-onnx
```

Export output directories must be new. The initial graph uses a fixed batch
size (default 1) and sequence length (default 256); inputs must be left-padded
without truncating. The float32 graph/weights take about 3 GB and are gitignored.
The verification example compares Julia's ONNX scores with three recorded
PyTorch references, including Japanese text. It consumes prepared token IDs,
so Python is not involved in verification or inference.

Use `backend = load_export("artifacts/jeff-0.8b-onnx")` to load the graph with
the checkpoint's temperature and option limit automatically. Call `close(backend)`
after use.

The reference source checkout is intentionally ignored by Git. To set it up,
run `git clone https://github.com/firelex/jeff extern/jeff`, then create its Python
environment according to its README. `tools/export_onnx.jl` uses PythonCall.jl
with that environment (`JULIA_PYTHONCALL_EXE` overrides the interpreter).
PythonCall is a tools dependency, not a dependency of Julia model inference.

