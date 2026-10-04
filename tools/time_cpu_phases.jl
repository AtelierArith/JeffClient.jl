using JeffClient, LinearAlgebra, Statistics
using QwenDecisionCore
import JSON
if QwenDecisionCore.cpu_setting(:accelerate)
    import AppleAccelerate
end
if QwenDecisionCore.cpu_setting(:portable_vector_math)
    import LoopVectorization
    @assert Base.get_extension(JeffClient, :JeffClientLoopVectorizationExt) !== nothing
end

function phase_layer(
    layer,
    hidden,
    mask,
    cfg,
    mlp_buffers,
    delta_buffers,
    final,
    delta_projections = nothing,
)
    pre = @timed QwenDecisionCore.native_rms(hidden, layer.input_norm, cfg.eps)
    attention = @timed if layer.attention.kind == :full
        final ?
        QwenDecisionCore.cpu_final_full_attention(layer.attention, pre.value, mask, cfg) : QwenDecisionCore.full_attention(layer.attention, pre.value, mask, cfg)
    else
        QwenDecisionCore.delta_attention(
            layer.attention,
            pre.value,
            mask,
            cfg,
            delta_buffers,
            delta_projections,
        )
    end
    post = @timed if final
        residual = @views hidden[:, end:end] .+ attention.value[:, end:end]
        (residual, QwenDecisionCore.native_rms(residual, layer.post_norm, cfg.eps))
    else
        QwenDecisionCore.native_residual_rms(hidden, attention.value, layer.post_norm, cfg.eps)
    end
    residual, normalized = post.value
    gate, up = mlp_buffers
    projections = @timed begin
        QwenDecisionCore.cpu_projection!(gate, layer.mlp.gate, normalized)
        QwenDecisionCore.cpu_projection!(up, layer.mlp.up, normalized)
    end
    activation = @timed QwenDecisionCore.cpu_owned_mlp_gate!(gate, up)
    fused = QwenDecisionCore.cpu_setting(:mlp_residual_fusion)
    down = @timed if fused
        QwenDecisionCore.cpu_projection!(residual, layer.mlp.down, activation.value, 1.0f0)
    else
        QwenDecisionCore.native_linear(layer.mlp.down, activation.value)
    end
    added = @timed if fused
        residual
    else
        QwenDecisionCore.native_residual_add!(residual, down.value)
    end
    phases = (pre, attention, post, projections, activation, down, added)
    return added.value, [p.time for p in phases], [p.bytes for p in phases]
end

function phase_pass(backend, ids, mask)
    return QwenDecisionCore.cpu_projection_scope() do
        phase_pass_unscoped(backend, ids, mask)
    end
end

function phase_pass_unscoped(backend, ids, mask)
    cfg = backend.backbone.config
    hidden = QwenDecisionCore.native_gather(backend.backbone.embedding, ids)
    workspace = QwenDecisionCore.cpu_mlp_workspace(backend.backbone.layers, length(ids))
    delta = QwenDecisionCore.cpu_delta_workspace(cfg, length(ids))
    projections = QwenDecisionCore.cpu_delta_projection_workspace(cfg, length(ids))
    records = []
    for (i, layer) in enumerate(backend.backbone.layers)
        final =
            i == length(backend.backbone.layers) &&
            QwenDecisionCore.cpu_setting(:final_token_only)
        buffers = final ? workspace.final : workspace.full
        hidden, times, bytes =
            phase_layer(layer, hidden, mask, cfg, buffers, delta, final, projections)
        push!(
            records,
            (layer = i, kind = layer.attention.kind, ms = times .* 1000, bytes = bytes),
        )
    end
    scores = QwenDecisionCore.native_linear(
        backend.readout,
        QwenDecisionCore.native_rms(
            hidden[:, end:end],
            backend.backbone.final_norm,
            cfg.eps,
        ),
    )
    return scores, records
end

function main()
    length(ARGS) == 2 || error("Usage: time_cpu_phases.jl CHECKPOINT REFERENCE")
    QwenDecisionCore.initialize_cpu!()
    backend = NativeBackend(ARGS[1])
    sample = only(JSON.parsefile(ARGS[2]))
    ids = Int64.(only(sample["inputs"]["input_ids"]))
    mask = Int64.(only(sample["inputs"]["attention_mask"]))
    start = QwenDecisionCore.cpu_setting(:trim_padding) ? findfirst(!iszero, mask) : 1
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
