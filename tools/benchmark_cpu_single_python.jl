# Isolated diagnostic wrapper; production CPU defaults are unchanged.
source = read(joinpath(@__DIR__, "benchmark_original.jl"), String)
source = replace(
    source,
    "include(\"python_env.jl\")" =>
        "include(" * repr(joinpath(@__DIR__, "python_env.jl")) * ")",
    "batch = PreparedBatch" => "count = json.loads((Path(checkpoint) / \"decision_config.json\").read_text())[\"max_options\"]\nbatch = PreparedBatch",
    "cpu_threads=8" => "cpu_threads=1",
    "import torch\n" => "import torch\nimport gc\ntorch.set_num_interop_threads(1)\n",
    "\"samples\": len(times)," => "\"times_ms\": times, \"samples\": len(times),",
    "    times = []" => "    gc.collect()\n    times = []",
    "\"cpu_threads\": torch.get_num_threads()," => "\"cpu_threads\": torch.get_num_threads(), \"interop_threads\": torch.get_num_interop_threads(),",
)
include_string(Main, source, abspath(joinpath(@__DIR__, "benchmark_original.jl")))
