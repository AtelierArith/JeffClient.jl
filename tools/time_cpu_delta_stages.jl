using JeffClient, LinearAlgebra, Statistics
import JSON, LoopVectorization
@assert Base.get_extension(JeffClient, :JeffClientLoopVectorizationExt) !== nothing

function delta_stages(attention, x, mask, cfg, workspace)
    n = size(x, 2)
    masked = @timed x .* reshape(Float32.(mask), 1, :)
    projection = @timed JeffClient.native_linear(attention.qkv, masked.value)
    convolution = @timed begin
        raw = zeros(Float32, size(projection.value))
        JeffClient.cpu_convolution!(raw, projection.value, attention.conv)
        raw
    end
    raw = convolution.value
    domain = (;
        minimum = minimum(raw),
        maximum = maximum(raw),
        low = count(x -> x <= -20.0f0, raw),
        high = count(x -> x >= 80.0f0, raw),
        tiny = count(x -> abs(x) <= 1.0f-12, raw),
    )
    activation = @timed JeffClient.native_cpu_silu!(raw, projection.value)
    prepared = @timed begin
        mixed = activation.value
        width = cfg.key_dim * cfg.key_heads
        q = reshape(copy(@view(mixed[1:width, :])), cfg.key_dim, cfg.key_heads, n)
        k = reshape(copy(@view(mixed[(width+1):2width, :])), cfg.key_dim, cfg.key_heads, n)
        v = reshape(@view(mixed[(2width+1):end, :]), cfg.value_dim, cfg.value_heads, n)
        JeffClient.cpu_normalize_delta_heads!(q, sqrt(Float32(cfg.key_dim)))
        JeffClient.cpu_normalize_delta_heads!(k, 1.0f0)
        (q, k, v)
    end
    gating = @timed begin
        z = reshape(
            JeffClient.native_linear(attention.z, masked.value),
            cfg.value_dim,
            cfg.value_heads,
            n,
        )
        beta = JeffClient.native_sigmoid.(
            JeffClient.native_linear(attention.b, masked.value),
        )
        decay =
            attention.a_decay .* JeffClient.native_softplus.(
                JeffClient.native_linear(attention.a, masked.value) .+ attention.dt_bias,
            )
        (z, beta, decay)
    end
    q, k, v = prepared.value
    z, beta, decay = gating.value
    out = Matrix{Float32}(undef, cfg.value_dim*cfg.value_heads, n)
    heads = @timed begin
        workers = JeffClient.cpu_delta_workers(cfg)
        @sync for worker = 1:workers
            Threads.@spawn JeffClient.cpu_delta_heads!(
                worker:workers:cfg.value_heads,
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
                false,
                workspace[worker],
            )
        end
    end
    output = @timed JeffClient.native_linear(attention.out, out)
    stages = (masked, projection, convolution, activation, prepared, gating, heads, output)
    return output.value, [p.time*1000 for p in stages], [p.bytes for p in stages], domain
end

function pass(backend, ids, mask)
    return JeffClient.cpu_projection_scope() do
        cfg = backend.config
        hidden = JeffClient.native_gather(backend.embedding, ids)
        workspace = JeffClient.cpu_mlp_workspace(backend.layers, length(ids))
        delta_workspace = JeffClient.cpu_delta_workspace(cfg, length(ids))
        records = []
        for (index, layer) in enumerate(backend.layers)
            final = index == length(backend.layers)
            normalized = JeffClient.native_rms(hidden, layer.input_norm, cfg.eps)
            if layer.attention.kind == :delta
                mixed, ms, bytes, domain =
                    delta_stages(layer.attention, normalized, mask, cfg, delta_workspace)
                push!(records, (; layer = index, ms, bytes, domain))
            else
                mixed =
                    final ?
                    JeffClient.cpu_final_full_attention(
                        layer.attention,
                        normalized,
                        mask,
                        cfg,
                    ) : JeffClient.full_attention(layer.attention, normalized, mask, cfg)
            end
            residual, normalized = if final
                residual = hidden[:, end:end] .+ mixed[:, end:end]
                (residual, JeffClient.native_rms(residual, layer.post_norm, cfg.eps))
            else
                JeffClient.native_residual_rms(hidden, mixed, layer.post_norm, cfg.eps)
            end
            buffers = final ? workspace.final : workspace.full
            hidden = JeffClient.native_residual_add!(
                residual,
                JeffClient.native_mlp(layer.mlp, normalized, buffers),
            )
        end
        scores = JeffClient.native_linear(
            backend.readout,
            JeffClient.native_rms(hidden[:, end:end], backend.final_norm, cfg.eps),
        )
        return scores, records
    end
end

function main()
    JeffClient.cpu_setting(:delta_projection_workspace) && error(
        "This stage diagnostic requires projection workspace disabled; use time_cpu_phases.jl to profile the workspace path.",
    )
    length(ARGS) == 2 || error("Usage: time_cpu_delta_stages.jl MODEL REFERENCE")
    BLAS.set_num_threads(1)
    backend = NativeBackend(ARGS[1])
    sample = only(JSON.parsefile(ARGS[2]))
    ids = Int64.(only(sample["inputs"]["input_ids"]))
    mask = Int64.(only(sample["inputs"]["attention_mask"]))
    start = findfirst(!iszero, mask)
    ids, mask = ids[start:end], mask[start:end]
    expected = Float32.(only(sample["logits"]))
    pass(backend, ids, mask)
    for repeat = 1:5
        measured = @timed pass(backend, ids, mask)
        scores, layers = measured.value
        isapprox(vec(scores), expected; atol = 2e-4, rtol = 2e-4) ||
            error("Reference mismatch")
        println(
            JSON.json((;
                repeat,
                total_ms = measured.time*1000,
                gc_ms = measured.gctime*1000,
                stages = [
                    "mask",
                    "qkv",
                    "convolution",
                    "silu",
                    "qk_prepare",
                    "z_beta_decay",
                    "state_rms_gate",
                    "output_projection",
                ],
                layers,
            )),
        )
        flush(stdout)
    end
end
JeffClient.with_cpu_settings(:delta_projection_workspace => false) do
    main()
end
