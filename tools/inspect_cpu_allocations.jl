# Run in an isolated environment containing AllocCheck and AppleAccelerate.
# AllocCheck 0.2.6 and Metal 1.11.1 require incompatible GPUCompiler versions.
push!(LOAD_PATH, dirname(@__DIR__))
using AllocCheck, JeffClient, LinearAlgebra
import JSON
get(ENV, "JEFF_CPU_ACCELERATE", "0") == "1" && (@eval import AppleAccelerate)

function main()
    length(ARGS) == 2 ||
        error("Usage: inspect_cpu_allocations.jl CHECKPOINT REFERENCE_JSON")
    backend = NativeBackend(ARGS[1])
    document = JSON.parsefile(ARGS[2])
    sample = first(document isa AbstractDict ? document["cases"] : document)
    matrix(rows) = reduce(vcat, [permutedims(Int64.(row)) for row in rows])
    inputs = Dict(name => matrix(rows) for (name, rows) in sample["inputs"])
    layer = first(backend.layers)
    x = zeros(Float32, backend.config.hidden, 101)
    gate = zeros(Float32, size(layer.mlp.gate, 2), size(x, 2))
    up = similar(gate)
    cfg = backend.config
    attention = first(l.attention for l in backend.layers if l.attention.kind == :delta)
    q = zeros(Float32, cfg.key_dim, cfg.key_heads, 101)
    k = copy(q)
    v = zeros(Float32, cfg.value_dim, cfg.value_heads, 101)
    z = copy(v)
    out = zeros(Float32, cfg.value_dim * cfg.value_heads, 101)
    beta = fill(0.5f0, cfg.value_heads, 101)
    decay = fill(-0.1f0, cfg.value_heads, 101)
    owned = JeffClient.cpu_delta_worker_workspace(cfg, 101)
    targets = (
        projection = (mul!, (gate, transpose(layer.mlp.gate), x)),
        owned_gate = (JeffClient.cpu_owned_mlp_gate!, (gate, up)),
        worker_workspace = (JeffClient.cpu_delta_worker_workspace, (backend.config, 101)),
        recurrent_kernel = (
            JeffClient.cpu_delta_recurrent_heads!,
            (
                1:cfg.value_heads,
                out,
                q,
                k,
                v,
                beta,
                decay,
                z,
                attention,
                cfg,
                cfg.value_heads ÷ cfg.key_heads,
                owned,
            ),
        ),
        logits = (logits, (backend, inputs)),
    )
    println("Julia ", VERSION, "; AllocCheck ", pkgversion(AllocCheck))
    for name in keys(targets)
        f, args = getproperty(targets, name)
        println("\nTarget: ", name)
        flush(stdout)
        errors = AllocCheck.check_allocs(f, Tuple{map(typeof, args)...})
        println("Static findings: ", length(errors))
        for error in Iterators.take(errors, 30)
            show(stdout, MIME"text/plain"(), error)
            println()
        end
        f(args...)
        println("Warmed Julia heap bytes: ", @allocated f(args...))
        flush(stdout)
    end
end

main()
