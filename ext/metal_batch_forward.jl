function batched_delta_attention(attention, normalized, mask, cfg, sequence_length)
    masked = delta_masked_input(normalized, metal_device_mask(normalized, mask))
    columns = size(masked, 2)
    mixed = batched_causal_depthwise(
        JeffClient.native_linear(attention.qkv, masked),
        attention.conv,
        sequence_length,
    )
    query, key = packed_qk_pair(mixed, cfg, columns)
    beta, decay = delta_gates(
        JeffClient.native_linear(attention.b, masked),
        JeffClient.native_linear(attention.a, masked),
        attention.a_decay,
        attention.dt_bias,
    )
    output = batched_delta_recurrent(query, key, mixed, beta, decay, cfg, sequence_length)
    z = reshape(
        JeffClient.native_linear(attention.z, masked),
        cfg.value_dim,
        cfg.value_heads,
        columns,
    )
    gated = rms_silu_gate(
        output,
        z,
        attention.norm,
        cfg.eps,
        (cfg.value_dim * cfg.value_heads, columns),
    )
    return JeffClient.native_linear(attention.out, gated)
end

function batched_readout_columns_kernel!(
    output,
    residual,
    mlp,
    sequence_length,
    ::Val{ADD},
) where {ADD}
    index = Metal.thread_position_in_grid_2d()
    row, sample = Int32(index.x), Int32(index.y)
    if row <= size(output, 1) && sample <= size(output, 2)
        column = sample * sequence_length
        @inbounds output[row, sample] =
            residual[row, column] + (ADD ? mlp[row, column] : 0.0f0)
    end
    return
end

function batched_readout_columns(residual, mlp, sequence_length, add)
    output = pooled_array(Float32, (size(residual, 1), size(residual, 2) ÷ sequence_length))
    launch_cached_kernel!(
        batched_readout_columns_kernel!,
        output,
        residual,
        mlp,
        Int32(sequence_length),
        add;
        threads = (64, 4),
        groups = (cld(size(output, 1), 64), cld(size(output, 2), 4)),
    )
    return output
end

function batched_hidden_forward(hidden, layers, mask, final_norm, cfg, sequence_length)
    if isempty(layers)
        return JeffClient.native_rms(
            batched_readout_columns(hidden, hidden, sequence_length, Val(false)),
            final_norm,
            cfg.eps,
        )
    end
    normalized = JeffClient.native_rms(hidden, first(layers).input_norm, cfg.eps)
    for index in eachindex(layers)
        layer = layers[index]
        mixed =
            layer.attention.kind == :full ?
            batched_full_attention(
                layer.attention,
                normalized,
                mask,
                cfg,
                sequence_length,
            ) :
            batched_delta_attention(layer.attention, normalized, mask, cfg, sequence_length)
        residual, post_normalized =
            JeffClient.native_residual_rms(hidden, mixed, layer.post_norm, cfg.eps)
        mlp = JeffClient.native_mlp(layer.mlp, post_normalized)
        if index == lastindex(layers)
            last_columns =
                batched_readout_columns(residual, mlp, sequence_length, Val(true))
            return JeffClient.native_rms(last_columns, final_norm, cfg.eps)
        end
        hidden, normalized =
            residual_input_rms!(residual, mlp, layers[index+1].input_norm, cfg.eps)
    end
    error("Batch layer traversal did not reach readout.")
end

function JeffClient.native_batch_logits(
    backend::JeffClient.NativeBackend{<:Metal.MtlMatrix{Float32}},
    ids,
    mask,
)
    get(ENV, "JEFF_METAL_BATCHED", "0") == "1" && size(ids, 1) > 1 || return nothing
    cfg = backend.config
    cfg.key_dim <= 256 && cfg.head_dim <= 4096 && size(backend.embedding, 1) <= 4096 ||
        return nothing
    first_token = minimum(
        JeffClient.native_sequence_start(backend.embedding, mask, row) for
        row in axes(ids, 1)
    )
    sequence_length = size(ids, 2) - first_token + 1
    # Initial batch prototype uses one workspace. Release inactive row-shape banks
    # before switching modes, while preserving an active enclosing scope.
    current = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    bank = get(task_local_storage(), SHAPE_WORKSPACE_KEY, nothing)
    bank isa ShapeWorkspaces &&
        !(current isa ForwardWorkspace && current.active) &&
        clear_forward_workspace!()
    return JeffClient.native_forward_scope(backend.embedding) do
        flat_ids = vec(permutedims(ids[:, first_token:end]))
        flat_mask = vec(permutedims(mask[:, first_token:end]))
        hidden = JeffClient.native_gather(backend.embedding, flat_ids)
        prepared = JeffClient.native_prepare_mask(hidden, flat_mask)
        result = batched_hidden_forward(
            hidden,
            backend.layers,
            prepared,
            backend.final_norm,
            cfg,
            sequence_length,
        )
        permutedims(
            JeffClient.native_host(JeffClient.native_linear(backend.readout, result)),
        )
    end
end
