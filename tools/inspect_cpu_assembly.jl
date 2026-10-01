using JeffClient, LinearAlgebra, InteractiveUtils

function main()
    backend = NativeBackend(only(ARGS))
    attention =
        first(layer.attention for layer in backend.layers if layer.attention.kind == :delta)
    cfg = backend.config
    tokens = 101
    x = ones(Float32, cfg.hidden, tokens)
    gate = ones(Float32, size(first(backend.layers).mlp.gate, 2), tokens)
    up = copy(gate)
    output = similar(gate)
    q = fill(inv(sqrt(Float32(cfg.key_dim))), cfg.key_dim, cfg.key_heads, tokens)
    k = copy(q)
    v = ones(Float32, cfg.value_dim, cfg.value_heads, tokens)
    z = copy(v)
    beta = fill(0.5f0, cfg.value_heads, tokens)
    decay = fill(-0.1f0, cfg.value_heads, tokens)
    result = zeros(Float32, cfg.value_dim * cfg.value_heads, tokens)
    owned = JeffClient.cpu_delta_worker_workspace(cfg, tokens)
    targets = (
        silu = (JeffClient.native_cpu_silu!, (gate,)),
        mlp_gate = (JeffClient.cpu_owned_mlp_gate!, (gate, up)),
        scalar_gate = (JeffClient.native_mlp_gate!, (gate, up)),
        projection_block = (
            JeffClient.cpu_projection_block!,
            (output, first(backend.layers).mlp.gate, x, 1:(size(output, 1)÷8)),
        ),
        recurrent = (
            JeffClient.cpu_delta_recurrent_heads!,
            (
                1:1,
                result,
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
    )
    for name in keys(targets)
        f, args = getproperty(targets, name)
        f(args...)
        println("\n=== ", name, " LLVM ===")
        @code_llvm debuginfo=:none optimize=true f(args...)
        println("\n=== ", name, " NATIVE ===")
        @code_native debuginfo=:none syntax=:intel f(args...)
    end
end

main()
