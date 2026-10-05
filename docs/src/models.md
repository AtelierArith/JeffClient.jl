# Models

```julia
using JeffClient

checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990")
# Reuse the cache with network access disabled:
checkpoint = resolve_checkpoint("mstrasser/Jeff-Qwen3.5-0.8B";
    revision="0f212b3e72acb4dde3f7da61e925d6ab7f819990", offline=true)
```

`resolve_checkpoint` returns a local directory containing the backbone
(`config.json`, `model.safetensors`) and Jeff's decision files
(`decision_config.json`, `readout.safetensors`). It checks the existing Hugging
Face cache first, then QwenDecisionCore's Scratch.jl cache under your Julia
depot, and downloads only missing files. Tokenizer, template and license files
are fetched too; other assets are skipped. Pass `cache_dir="..."` to choose
another cache root. `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN` and
`HF_HUB_OFFLINE` are respected.

`NativeBackend` accepts either a repository id or a local checkpoint
directory; a local directory is validated and used as is.
