using JeffClient

length(ARGS) <= 2 ||
    error("Usage: julia --project tools/download_checkpoint.jl [org/model [revision]]")
repo = isempty(ARGS) ? "mstrasser/Jeff-Qwen3.5-0.8B" : ARGS[1]
revision =
    length(ARGS) == 2 ? ARGS[2] :
    (
        repo == "mstrasser/Jeff-Qwen3.5-0.8B" ? "0f212b3e72acb4dde3f7da61e925d6ab7f819990" :
        "main"
    )
println(resolve_checkpoint(repo; revision))
