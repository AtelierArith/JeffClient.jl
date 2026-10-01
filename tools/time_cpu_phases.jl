using JeffClient, LinearAlgebra, Statistics
import JSON
if get(ENV, "JEFF_CPU_ACCELERATE", "0") == "1"
    import AppleAccelerate
end
if get(ENV, "JEFF_CPU_PORTABLE_VECTOR_MATH", "0") == "1"
    import LoopVectorization
    @assert Base.get_extension(JeffClient, :JeffClientLoopVectorizationExt) !== nothing
end

function phase_layer(layer, hidden, mask, cfg, mlp_buffers, delta_buffers, final)
    pre = @timed JeffClient.native_rms(hidden, layer.input_norm, cfg.eps)
    attention = @timed if layer.attention.kind == :full
        final ?
        JeffClient.cpu_final_full_attention(layer.attention, pre.value, mask, cfg) :
        JeffClient.full_attention(layer.attention, pre.value, mask, cfg)
    else
        JeffClient.delta_attention(layer.attention, pre.value, mask, cfg, delta_buffers)
    end
    post = @timed if final
        residual = @views hidden[:, end:end] .+ attention.value[:, end:end]
        (residual, JeffClient.native_rms(residual, layer.post_norm, cfg.eps))
    else
        JeffClient.native_residual_rms(hidden, attention.value, layer.post_norm, cfg.eps)
    end
    residual, normalized = post.value
    gate, up = mlp_buffers
    projections = @timed begin
        JeffClient.cpu_projection!(gate, layer.mlp.gate, normalized)
        JeffClient.cpu_projection!(up, layer.mlp.up, normalized)
    end
    activation = @timed JeffClient.cpu_owned_mlp_gate!(gate, up)
    down = @timed JeffClient.native_linear(layer.mlp.down, activation.value)
    added = @timed JeffClient.native_residual_add!(residual, down.value)
    phases = (pre, attention, post, projections, activation, down, added)
    return added.value, [p.time for p in phases], [p.bytes for p in phases]
end

function phase_pass(backend, ids, mask)
    return JeffClient.cpu_projection_scope() do
        phase_pass_unscoped(backend, ids, mask)
    end
end

function phase_pass_unscoped(backend, ids, mask)
    cfg = backend.config
    hidden = JeffClient.native_gather(backend.embedding, ids)
    workspace = JeffClient.cpu_mlp_workspace(backend.layers, length(ids))
    delta = JeffClient.cpu_delta_workspace(cfg, length(ids))
    records = []
    for (i, layer) in enumerate(backend.layers)
        final = i == length(backend.layers)
        buffers = final ? workspace.final : workspace.full
        hidden, times, bytes = phase_layer(layer, hidden, mask, cfg, buffers, delta, final)
        push!(
            records,
            (layer = i, kind = layer.attention.kind, ms = times .* 1000, bytes = bytes),
        )
    end
    scores = JeffClient.native_linear(
        backend.readout,
        JeffClient.native_rms(hidden[:, end:end], backend.final_norm, cfg.eps),
    )
    return scores, records
end

function main()
    length(ARGS) == 2 || error("Usage: time_cpu_phases.jl CHECKPOINT REFERENCE")
    BLAS.set_num_threads(parse(Int, get(ENV, "JEFF_BLAS_THREADS", "8")))
    backend = NativeBackend(ARGS[1])
    sample = only(JSON.parsefile(ARGS[2]))
    ids = Int64.(only(sample["inputs"]["input_ids"]))
    mask = Int64.(only(sample["inputs"]["attention_mask"]))
    start = findfirst(!iszero, mask)
    ids, mask = ids[start:end], mask[start:end]
    expected = Float32.(only(sample["logits"]))
    phase_pass(backend, ids, mask)
    for pass = 1:5
        measured = @timed phase_pass(backend, ids, mask)
        scores, records = measured.value
        isapprox(vec(scores), expected; atol = 2e-4, rtol = 2e-4) ||
            error("Reference mismatch")
        println(
            JSON.json((
                pass = pass,
                total_ms = measured.time*1000,
                gc_ms = measured.gctime*1000,
                phases = [
                    "pre_rms",
                    "attention",
                    "post_rms",
                    "gate_up_projection",
                    "gate_activation",
                    "down_projection",
                    "residual_add",
                ],
                layers = records,
            )),
        )
        flush(stdout)
    end
end

main()
