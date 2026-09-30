using BenchmarkTools
using JeffClient
using LinearAlgebra
import JSON
import Metal

function completed_call(f, args)
    result = f(args...)
    Metal.synchronize()
    return result
end

function measure_stage(name, f, args)
    completed_call(f, args)
    completed_call(f, args)
    GC.gc(true)
    trial = @benchmark completed_call($f, $args) samples=10 evals=1 seconds=30
    measured = BenchmarkTools.median(trial)
    return Dict(
        "stage" => name,
        "median_ms" => measured.time / 1e6,
        "julia_heap_bytes" => measured.memory,
        "julia_allocations" => measured.allocs,
    )
end

function mlp_projection(mlp, x)
    gate = JeffClient.native_linear(mlp.gate, x)
    up = JeffClient.native_linear(mlp.up, x)
    return JeffClient.native_linear(mlp.down, JeffClient.native_mlp_gate!(gate, up))
end

function packed_mlp_gate_kernel!(output, packed)
    index = Int32(Metal.thread_position_in_grid_1d())
    width = Int32(size(output, 1))
    if index <= length(output)
        row = rem(index - Int32(1), width) + Int32(1)
        column = div(index - Int32(1), width)
        source = row + Int32(2) * width * column
        @inbounds output[index] =
            JeffClient.native_silu(packed[source]) * packed[source+width]
    end
    return
end

function packed_mlp_projection(extension, packed_weight, down, x)
    packed = JeffClient.native_linear(packed_weight, x)
    gate = extension.pooled_array(Float32, (size(packed, 1) ÷ 2, size(x, 2)))
    extension.launch_cached_kernel!(
        packed_mlp_gate_kernel!,
        gate,
        packed;
        threads = 256,
        groups = cld(length(gate), 256),
    )
    return JeffClient.native_linear(down, gate)
end

function delta_input_projections(attention, x)
    return (
        JeffClient.native_linear(attention.qkv, x),
        JeffClient.native_linear(attention.z, x),
        JeffClient.native_linear(attention.b, x),
        JeffClient.native_linear(attention.a, x),
    )
end

function packed_qk_separate(extension, mixed, cfg, sequence_length)
    return (
        extension.packed_qk(mixed, cfg, sequence_length, 0, sqrt(Float32(cfg.key_dim))),
        extension.packed_qk(
            mixed,
            cfg,
            sequence_length,
            cfg.key_dim * cfg.key_heads,
            1.0f0,
        ),
    )
end

function delta_gates_separate(b, a, a_decay, dt_bias)
    return JeffClient.native_sigmoid.(b),
    a_decay .* JeffClient.native_softplus.(a .+ dt_bias)
end

function delta_recurrent_stage(
    extension,
    query,
    key,
    mixed,
    beta,
    decay,
    cfg,
    ::Val{ROWS} = Val(8),
    ::Val{PRECOMPUTED} = Val(false),
) where {ROWS,PRECOMPUTED}
    sequence_length = size(mixed, 2)
    output =
        extension.pooled_array(Float32, (cfg.value_dim, cfg.value_heads, sequence_length))
    rows = ROWS
    Metal.@metal threads=(32, rows) groups=(cld(cfg.value_dim, rows), cfg.value_heads) extension.delta_recurrent_kernel!(
        output,
        query,
        key,
        mixed,
        beta,
        decay,
        Int32(cfg.key_dim),
        Int32(cfg.value_dim),
        Int32(2cfg.key_dim * cfg.key_heads),
        Int32(cfg.value_heads ÷ cfg.key_heads),
        Int32(sequence_length),
        Val(cld(cfg.key_dim, 32)),
        Val(rows),
        Val(PRECOMPUTED),
    )
    return output
end

function main()
    length(ARGS) in (2, 3) || error(
        "Usage: julia --project=tools tools/benchmark_stages.jl CHECKPOINT REFERENCE_JSON [OUTPUT_JSON]",
    )
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    BLAS.set_num_threads(8)
    references = JSON.parsefile(ARGS[2])
    cases = references isa AbstractDict ? references["cases"] : references
    sample = first(cases)
    ids = Int64.(first(sample["inputs"]["input_ids"]))
    mask = Int64.(first(sample["inputs"]["attention_mask"]))
    backend = NativeBackend(ARGS[1]; device = :metal)
    hidden = JeffClient.native_gather(backend.embedding, ids)
    prepared_mask = JeffClient.native_prepare_mask(hidden, mask)
    delta = first(layer for layer in backend.layers if layer.attention.kind == :delta)
    full = first(layer for layer in backend.layers if layer.attention.kind == :full)
    mixed = JeffClient.native_linear(delta.attention.qkv, hidden)
    extension = Base.get_extension(JeffClient, :JeffClientMetalExt)
    cfg = backend.config
    masked = hidden .* reshape(prepared_mask.device, 1, :)
    delta_mixed = JeffClient.causal_depthwise(
        JeffClient.native_linear(delta.attention.qkv, masked),
        delta.attention.conv,
    )
    query =
        extension.packed_qk(delta_mixed, cfg, length(ids), 0, sqrt(Float32(cfg.key_dim)))
    key = extension.packed_qk(
        delta_mixed,
        cfg,
        length(ids),
        cfg.key_dim * cfg.key_heads,
        1.0f0,
    )
    b_projection = JeffClient.native_linear(delta.attention.b, masked)
    a_projection = JeffClient.native_linear(delta.attention.a, masked)
    beta, decay = extension.delta_gates(
        b_projection,
        a_projection,
        delta.attention.a_decay,
        delta.attention.dt_bias,
    )
    gate = reshape(
        JeffClient.native_linear(delta.attention.z, masked),
        cfg.value_dim,
        cfg.value_heads,
        length(ids),
    )
    recurrent = delta_recurrent_stage(extension, query, key, delta_mixed, beta, decay, cfg)
    reference_recurrent = Array(recurrent)
    decay_factor = exp.(decay)
    factor_recurrent = Array(
        delta_recurrent_stage(
            extension,
            query,
            key,
            delta_mixed,
            beta,
            decay_factor,
            cfg,
            Val(8),
            Val(true),
        ),
    )
    isapprox(factor_recurrent, reference_recurrent; atol = 2.0f-5, rtol = 2.0f-5) ||
        error("Precomputed decay factor mismatch.")
    for rows in (1, 4, 16)
        probe = Array(
            delta_recurrent_stage(
                extension,
                query,
                key,
                delta_mixed,
                beta,
                decay,
                cfg,
                Val(rows),
            ),
        )
        isapprox(probe, reference_recurrent; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("Recurrent threadgroup configuration mismatch.")
    end
    # Pack once outside timing; this diagnostic retains an extra copy of weights.
    packed_mlp_weight = Metal.MtlArray(hcat(Array(delta.mlp.gate), Array(delta.mlp.up)))
    ordinary_mlp = Array(completed_call(mlp_projection, (delta.mlp, hidden)))
    packed_mlp = Array(
        completed_call(
            packed_mlp_projection,
            (extension, packed_mlp_weight, delta.mlp.down, hidden),
        ),
    )
    isapprox(packed_mlp, ordinary_mlp; atol = 2.0f-4, rtol = 2.0f-4) ||
        error("Packed MLP projection mismatch.")
    # Isolated, synchronized stage costs diagnose kernels/submission overhead.
    # They are not additive: complete forwards batch operations differently.
    results = [
        measure_stage(name, f, args) for (name, f, args) in (
            (
                "DeltaNet attention",
                JeffClient.delta_attention,
                (delta.attention, hidden, prepared_mask, backend.config),
            ),
            (
                "full attention",
                JeffClient.full_attention,
                (full.attention, hidden, prepared_mask, backend.config),
            ),
            ("MLP projections", mlp_projection, (delta.mlp, hidden)),
            (
                "MLP packed gate/up projections",
                packed_mlp_projection,
                (extension, packed_mlp_weight, delta.mlp.down, hidden),
            ),
            (
                "RMS normalization",
                JeffClient.native_rms,
                (hidden, delta.input_norm, backend.config.eps),
            ),
            (
                "causal depthwise convolution",
                JeffClient.causal_depthwise,
                (mixed, delta.attention.conv),
            ),
            ("QKV projection", JeffClient.native_linear, (delta.attention.qkv, hidden)),
            (
                "DeltaNet input projections",
                delta_input_projections,
                (delta.attention, masked),
            ),
            (
                "DeltaNet recurrent",
                delta_recurrent_stage,
                (extension, query, key, delta_mixed, beta, decay, cfg),
            ),
            (
                "DeltaNet RMS/SiLU gate",
                extension.rms_silu_gate,
                (recurrent, gate, delta.attention.norm, cfg.eps),
            ),
        )
    ]
    append!(
        results,
        [
            measure_stage(
                "DeltaNet recurrent rows=$rows",
                delta_recurrent_stage,
                (extension, query, key, delta_mixed, beta, decay, cfg, Val(rows)),
            ) for rows in (1, 4, 16)
        ],
    )
    push!(
        results,
        measure_stage(
            "DeltaNet recurrent precomputed factor",
            delta_recurrent_stage,
            (
                extension,
                query,
                key,
                delta_mixed,
                beta,
                decay_factor,
                cfg,
                Val(8),
                Val(true),
            ),
        ),
    )
    push!(
        results,
        measure_stage(
            "DeltaNet packed Q/K separate",
            packed_qk_separate,
            (extension, delta_mixed, cfg, length(ids)),
        ),
    )
    push!(
        results,
        measure_stage(
            "DeltaNet packed Q/K pair",
            extension.packed_qk_pair,
            (delta_mixed, cfg, length(ids)),
        ),
    )
    for (name, f) in (
        ("DeltaNet beta/decay separate", delta_gates_separate),
        ("DeltaNet beta/decay fused", extension.delta_gates),
    )
        push!(
            results,
            measure_stage(
                name,
                f,
                (
                    b_projection,
                    a_projection,
                    delta.attention.a_decay,
                    delta.attention.dt_bias,
                ),
            ),
        )
    end
    report = Dict(
        "delta_gates_fused" => true,
        "device" => string(Metal.device().name),
        "julia_version" => string(VERSION),
        "metal_version" => string(pkgversion(Metal)),
        "sequence_length" => length(ids),
        "active_tokens" => count(!iszero, mask),
        "recurrent_baseline_rows" => 8,
        "recurrent_variant_rows" => [1, 4, 16],
        "timing_scope" => "isolated stages with GPU completion; not additive",
        "samples_per_stage" => 10,
        "stages" => results,
    )
    println(JSON.json(report, 2))
    if length(ARGS) == 3
        mkpath(dirname(abspath(ARGS[3])))
        write(ARGS[3], JSON.json(report, 2) * "\n")
    end
end

main()
