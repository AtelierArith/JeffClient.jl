# Original PyTorch GPU forward with CPU-prepared input upload included.
source = read(joinpath(@__DIR__, "benchmark_original.jl"), String)
source = replace(
    source,
    "include(\"python_env.jl\")" =>
        "include(" * repr(joinpath(@__DIR__, "python_env.jl")) * ")",
    "import torch\n" => "import torch\nimport gc\n",
    "dtype=torch.int64, device=device)" => "dtype=torch.int64, device='cpu')",
    "batch = PreparedBatch" => "count = json.loads((Path(checkpoint) / \"decision_config.json\").read_text())[\"max_options\"]\nbatch = PreparedBatch",
    "result = model(batch).cpu()" => "device_batch = PreparedBatch({k: v.to(device) for k, v in inputs.items()}, batch.counts, batch.input_tokens)\n    result = model(device_batch).cpu()",
    "\"samples\": len(times)," => "\"cpu_input_upload_included\": True, \"times_ms\": times, \"samples\": len(times),",
    "    times = []" => "    gc.collect()\n    times = []",
)
length(ARGS) >= 2 && ARGS[2] == "mps-f32" ||
    error("Use mps-f32 for the equal-precision GPU comparison.")
include_string(Main, source, joinpath(@__DIR__, "benchmark_original.jl"))
