using JeffClient
using QwenDecisionCore
using LinearAlgebra
import Metal

include("verify_metal_command_buffers.jl")

function kernel_handle_probe!(output)
    index = Int32(Metal.thread_position_in_grid_1d())
    if index <= length(output)
        @inbounds output[index] = 1.0f0
    end
    return
end

function captured_kernel_handle_probe(value)
    return output -> begin
        index = Int32(Metal.thread_position_in_grid_1d())
        if index <= length(output)
            @inbounds output[index] = value
        end
        return
    end
end

function verify_batched_forward_workspace(extension)
    flags = (
        "JEFF_METAL_BATCHED",
        "JEFF_METAL_WORKSPACE",
        "JEFF_METAL_SHAPE_WORKSPACES",
        "JEFF_METAL_TRIM_PADDING",
    )
    previous = Dict(flag => get(ENV, flag, nothing) for flag in flags)
    try
        extension.clear_forward_workspace!()
        ENV["JEFF_METAL_BATCHED"] = "1"
        ENV["JEFF_METAL_WORKSPACE"] = "1"
        ENV["JEFF_METAL_SHAPE_WORKSPACES"] = "1"
        ENV["JEFF_METAL_TRIM_PADDING"] = "0"
        fixture = joinpath(@__DIR__, "..", "test", "fixtures", "native")
        cpu, gpu =
            NativeBackend(fixture; device = :cpu), NativeBackend(fixture; device = :metal)
        ids = reshape(mod.(collect(Int64, 0:17), size(gpu.backbone.embedding, 2)), 2, 9)
        masks =
            (ones(Int64, 2, 9), [0 0 1 1 0 1 1 0 1; 1 0 1 0 1 0 1 1 1], ones(Int64, 2, 9))
        previous_slots = nothing
        previous_tensors = nothing
        for mask in masks
            inputs = Dict("input_ids" => ids, "attention_mask" => mask)
            expected, actual = logits(cpu, inputs), logits(gpu, inputs)
            isapprox(actual, expected; atol = 2.0f-4, rtol = 2.0f-4) ||
                error("Batch workspace scores mismatch.")
            workspace = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
            !workspace.active || error("Batch workspace remained active after readback.")
            if previous_slots !== nothing
                length(previous_slots) == length(workspace.slots) &&
                all(a === b for (a, b) in zip(previous_slots, workspace.slots)) ||
                    error("Batch mask values replaced workspace slots.")
                all(
                    get(workspace.tensor_data, key, nothing) === value for
                    (key, value) in previous_tensors
                ) || error("Batch mask values replaced tensor-data bindings.")
            end
            previous_slots, previous_tensors =
                copy(workspace.slots), copy(workspace.tensor_data)
            GC.gc(true)
        end
        # Different batch sizes exercise slot replacement, followed by reuse.
        for batch in (3, 3, 2, 2)
            changed_ids = reshape(
                mod.(collect(Int64, 0:(batch*9-1)), size(gpu.backbone.embedding, 2)),
                batch,
                9,
            )
            inputs =
                Dict("input_ids" => changed_ids, "attention_mask" => ones(Int64, batch, 9))
            isapprox(
                logits(gpu, inputs),
                logits(cpu, inputs);
                atol = 2.0f-4,
                rtol = 2.0f-4,
            ) || error("Changed batch shape scores mismatch.")
            GC.gc(true)
        end
    finally
        extension.clear_forward_workspace!()
        for flag in flags
            previous[flag] === nothing ? delete!(ENV, flag) : (ENV[flag] = previous[flag])
        end
    end
    println(
        "Validated batch forward workspace slots/tensor-data, mask changes, batch sizes and GC.",
    )
end

function verify_batched_full_attention(extension)
    for (width, heads, kv_heads, rotary_dim, sequence_length, batch) in
        ((4, 2, 1, 4, 3, 3), (7, 3, 3, 2, 9, 2), (256, 8, 2, 64, 9, 2))
        hidden = 7
        cfg = (
            head_dim = width,
            heads = heads,
            kv_heads = kv_heads,
            rotary_dim = rotary_dim,
            rope_theta = 1.0f7,
            eps = 1.0f-6,
        )
        make_weight(rows, cols) =
            reshape(sin.(Float32.(1:(rows*cols))), rows, cols) .* 0.1f0
        attention = (
            q = make_weight(hidden, 2width * heads),
            k = make_weight(hidden, width * kv_heads),
            v = make_weight(hidden, width * kv_heads),
            out = make_weight(width * heads, hidden),
            q_norm = cos.(Float32.(1:width)) .* 0.1f0,
            k_norm = sin.(Float32.(1:width)) .* 0.1f0,
        )
        gpu_attention = map(Metal.MtlArray, attention)
        input = reshape(
            cos.(Float32.(1:(hidden*sequence_length*batch))),
            hidden,
            sequence_length * batch,
        )
        mask = Float32[
            token == sequence_length || (token + sample) % 3 == 0 ? 1 : 0 for
            sample = 1:batch for token = 1:sequence_length
        ]
        expected = hcat(
            [
                QwenDecisionCore.full_attention(
                    attention,
                    input[:, ((sample-1)*sequence_length+1):(sample*sequence_length)],
                    mask[((sample-1)*sequence_length+1):(sample*sequence_length)],
                    cfg,
                ) for sample = 1:batch
            ]...,
        )
        gpu_input, gpu_mask = Metal.MtlArray(input), Metal.MtlArray(mask)
        for pass = 1:2
            actual = Array(
                extension.batched_full_attention(
                    gpu_attention,
                    gpu_input,
                    gpu_mask,
                    cfg,
                    sequence_length,
                ),
            )
            isapprox(actual, expected; atol = 2.0f-4, rtol = 2.0f-4) ||
                error("Batched full attention mismatch: $width/$pass.")
            GC.gc(true)
        end
        permutation = vcat(
            [
                collect(((sample-1)*sequence_length+1):(sample*sequence_length)) for
                sample = batch:-1:1
            ]...,
        )
        copyto!(gpu_input, input[:, permutation])
        copyto!(gpu_mask, mask[permutation])
        actual = Array(
            extension.batched_full_attention(
                gpu_attention,
                gpu_input,
                gpu_mask,
                cfg,
                sequence_length,
            ),
        )
        isapprox(actual, expected[:, permutation]; atol = 2.0f-4, rtol = 2.0f-4) ||
            error("Batched attention sample permutation mismatch.")
    end
    println(
        "Validated batched full attention, grouped heads, RoPE, sample permutation and GC.",
    )
end

function verify_batched_masked_softmax(extension)
    for sequence_length in (1, 9, 65), heads in (1, 3), batch in (1, 2, 3)
        scores = reshape(
            sin.(Float32.(1:(sequence_length^2*heads*batch))),
            sequence_length,
            sequence_length,
            heads * batch,
        )
        mask = ones(Float32, sequence_length * batch)
        for sample = 1:batch, token = 1:sequence_length
            mask[(sample-1)*sequence_length+token] =
                token == sequence_length || (token + sample) % 3 == 0 ? 1.0f0 : 0.0f0
        end
        expected = similar(scores)
        scale = inv(sqrt(128.0f0))
        for sample = 1:batch, head = 1:heads, query = 1:sequence_length
            head_sample = (sample - 1) * heads + head
            values = Float32[
                key <= query && mask[(sample-1)*sequence_length+key] == 1.0f0 ?
                scores[key, query, head_sample] * scale : -floatmax(Float32) for
                key = 1:sequence_length
            ]
            values .= exp.(values .- maximum(values))
            expected[:, query, head_sample] = values ./ sum(values)
        end
        gpu_scores, gpu_mask = Metal.MtlArray(scores), Metal.MtlArray(mask)
        for pass = 1:2
            actual =
                Array(extension.batched_masked_softmax(gpu_scores, gpu_mask, 128, heads))
            isapprox(actual, expected; atol = 2.0f-6, rtol = 2.0f-5) ||
                error("Batched softmax mismatch: $sequence_length/$heads/$batch/$pass.")
            GC.gc(true)
        end
        if batch > 1
            mask[1:sequence_length] .= 1.0f0
            copyto!(gpu_mask, mask)
            actual =
                Array(extension.batched_masked_softmax(gpu_scores, gpu_mask, 128, heads))
            isapprox(
                actual[:, :, (heads+1):end],
                expected[:, :, (heads+1):end];
                atol = 2.0f-6,
                rtol = 2.0f-5,
            ) || error("Softmax used another sample's mask.")
        end
    end
    println("Validated batched causal/padding softmax masks and GC.")
end

function verify_batched_delta_recurrent(extension)
    for width in (7, 128, 256), sequence_length in (1, 9, 65), batch in (1, 2, 3)
        heads, value_heads, value_width = 2, 4, 9
        columns = sequence_length * batch
        channels = 2width * heads + value_width * value_heads
        mixed = reshape(sin.(Float32.(1:(channels*columns))), channels, columns)
        cfg = (
            key_dim = width,
            key_heads = heads,
            value_dim = value_width,
            value_heads = value_heads,
        )
        gpu_mixed = Metal.MtlArray(mixed)
        gpu_q, gpu_k = extension.packed_qk_pair(gpu_mixed, cfg, columns)
        q, k = Array(gpu_q), Array(gpu_k)
        beta = fill(0.5f0, value_heads, columns)
        decay = reshape(
            range(-12.0f0, 0.0f0; length = value_heads * columns),
            value_heads,
            columns,
        )
        gpu_beta, gpu_decay = Metal.MtlArray(beta), Metal.MtlArray(decay)
        expected = zeros(Float32, value_width, value_heads, columns)
        for sample = 1:batch, head = 1:value_heads
            key_head = cld(head, value_heads ÷ heads)
            state = zeros(Float32, width, value_width)
            for token = ((sample-1)*sequence_length+1):(sample*sequence_length)
                query, key = q[:, key_head, token], k[:, key_head, token]
                state .*= exp(decay[head, token])
                start = 2width * heads + (head - 1) * value_width
                v = mixed[(start+1):(start+value_width), token]
                correction = (v - transpose(state) * key) .* beta[head, token]
                state .+= key * transpose(correction)
                expected[:, head, token] = transpose(state) * query
            end
        end
        for pass = 1:2
            actual = Array(
                extension.batched_delta_recurrent(
                    gpu_q,
                    gpu_k,
                    gpu_mixed,
                    gpu_beta,
                    gpu_decay,
                    cfg,
                    sequence_length,
                ),
            )
            isapprox(actual, expected; atol = 2.0f-5, rtol = 2.0f-4) ||
                error("Batched recurrent mismatch: $width/$sequence_length/$batch/$pass.")
            GC.gc(true)
        end
        if batch > 1
            mixed[:, 1:sequence_length] .= 100.0f0
            copyto!(gpu_mixed, mixed)
            actual = Array(
                extension.batched_delta_recurrent(
                    gpu_q,
                    gpu_k,
                    gpu_mixed,
                    gpu_beta,
                    gpu_decay,
                    cfg,
                    sequence_length,
                ),
            )
            isapprox(
                actual[:, :, (sequence_length+1):end],
                expected[:, :, (sequence_length+1):end];
                atol = 2.0f-5,
                rtol = 2.0f-4,
            ) || error("Recurrent state crossed a sample boundary.")
        end
    end
    println("Validated batched recurrent state isolation across widths and GC.")
end

function verify_batched_causal_depthwise(extension)
    for channels in (7, 128),
        sequence_length in (1, 3, 9, 65),
        batch in (1, 2, 3),
        taps in (1, 4)

        columns = sequence_length * batch
        input = reshape(sin.(Float32.(1:(channels*columns))), channels, columns)
        weight = reshape(cos.(Float32.(1:(taps*channels))), taps, channels)
        expected = similar(input)
        for sample = 1:batch
            range = ((sample-1)*sequence_length+1):(sample*sequence_length)
            expected[:, range] = QwenDecisionCore.causal_depthwise(input[:, range], weight)
        end
        gpu_input, gpu_weight = Metal.MtlArray(input), Metal.MtlArray(weight)
        for pass = 1:2
            actual = Array(
                extension.batched_causal_depthwise(gpu_input, gpu_weight, sequence_length),
            )
            isapprox(actual, expected; atol = 2.0f-6, rtol = 2.0f-5) || error(
                "Batched causal convolution mismatch: $channels/$sequence_length/$batch/$taps/$pass.",
            )
            GC.gc(true)
        end
        if batch > 1
            # Poison the preceding sample; the following sample must be unchanged.
            input[:, 1:sequence_length] .= 100.0f0
            copyto!(gpu_input, input)
            actual = Array(
                extension.batched_causal_depthwise(gpu_input, gpu_weight, sequence_length),
            )
            isapprox(
                actual[:, (sequence_length+1):end],
                expected[:, (sequence_length+1):end];
                atol = 2.0f-6,
                rtol = 2.0f-5,
            ) || error("Convolution crossed a sample boundary.")
        end
    end
    println("Validated batched causal convolution across sample boundaries and GC.")
end

function verify_fused_delta_mask_workspace(extension)
    flags = (
        "JEFF_METAL_WORKSPACE",
        "JEFF_METAL_SHAPE_WORKSPACES",
        "JEFF_METAL_TRIM_PADDING",
        "JEFF_METAL_FUSED_DELTA_MASK",
    )
    previous = Dict(flag => get(ENV, flag, nothing) for flag in flags)
    try
        extension.clear_forward_workspace!()
        ENV["JEFF_METAL_WORKSPACE"] = "1"
        ENV["JEFF_METAL_SHAPE_WORKSPACES"] = "0"
        ENV["JEFF_METAL_TRIM_PADDING"] = "0"
        ENV["JEFF_METAL_FUSED_DELTA_MASK"] = "1"
        fixture = joinpath(@__DIR__, "..", "test", "fixtures", "native")
        cpu = NativeBackend(fixture; device = :cpu)
        gpu = NativeBackend(fixture; device = :metal)
        ids = reshape(mod.(collect(Int64, 0:8), size(gpu.backbone.embedding, 2)), 1, 9)
        previous_slots = nothing
        for mask in (
            ones(Int64, 1, 9),
            reshape(Int64[1, 0, 1, 0, 1, 0, 1, 0, 1], 1, 9),
            ones(Int64, 1, 9),
            reshape(Int64[0, 0, 1, 1, 0, 1, 0, 1, 1], 1, 9),
        )
            inputs = Dict("input_ids" => ids, "attention_mask" => mask)
            expected = logits(cpu, inputs)
            actual = logits(gpu, inputs)
            isapprox(actual, expected; atol = 2.0f-4, rtol = 2.0f-4) ||
                error("Fused delta mask workspace scores mismatch.")
            workspace = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
            if previous_slots !== nothing
                length(previous_slots) == length(workspace.slots) &&
                all(a === b for (a, b) in zip(previous_slots, workspace.slots)) ||
                    error("Mask values changed fused workspace slot identities.")
            end
            previous_slots = copy(workspace.slots)
            GC.gc(true)
        end
    finally
        extension.clear_forward_workspace!()
        for flag in flags
            previous[flag] === nothing ? delete!(ENV, flag) : (ENV[flag] = previous[flag])
        end
    end
    println(
        "Validated fused delta workspace slots across all-active/interior/prefix masks and GC.",
    )
end

function verify_weight_tensor_owner(extension)
    host_weight = reshape(sin.(Float32.(1:35)), 7, 5)
    weight = Metal.MtlArray(host_weight)
    key = objectid(weight)
    first_data = nothing
    for length in (1, 9, 65, 1)
        input = reshape(cos.(Float32.(1:7length)), 7, length)
        actual = Array(QwenDecisionCore.native_linear(weight, Metal.MtlArray(input)))
        isapprox(actual, transpose(host_weight) * input; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("Cached weight linear mismatch.")
        entry = extension.WEIGHT_TENSOR_CACHE[key]
        entry[1].value === weight || error("Weight cache owner mismatch.")
        if first_data === nothing
            first_data = entry[2]
        else
            entry[2] === first_data || error("Weight tensor-data was not reused.")
        end
        GC.gc(true)
    end
    return key
end

function main()
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    extension = Base.get_extension(JeffClient, :JeffClientMetalExt)
    if ARGS == ["batch"]
        verify_batched_causal_depthwise(extension)
        verify_batched_delta_recurrent(extension)
        verify_batched_masked_softmax(extension)
        verify_batched_full_attention(extension)
        verify_batched_forward_workspace(extension)
        return
    end
    isempty(ARGS) || error("Usage: verify_metal_primitives.jl [batch]")
    previous_trim = get(ENV, "JEFF_METAL_TRIM_PADDING", nothing)
    try
        reference = Metal.MtlArray(zeros(Float32, 1, 1))
        masks = [0 0 1 0 1; 1 0 1 0 1; 0 0 0 0 1; 1 1 1 1 1]
        ENV["JEFF_METAL_TRIM_PADDING"] = "1"
        for (row, start) in enumerate((3, 1, 5, 1))
            QwenDecisionCore.native_sequence_start(reference, masks, row) == start ||
                error("Leading padding start mismatch.")
            cpu_start = QwenDecisionCore.with_cpu_settings(:trim_padding => false) do
                QwenDecisionCore.native_sequence_start(zeros(Float32, 1, 1), masks, row)
            end
            cpu_start == 1 || error("Metal trimming flag changed the CPU policy.")
        end
        ENV["JEFF_METAL_TRIM_PADDING"] = "0"
        all(
            QwenDecisionCore.native_sequence_start(reference, masks, row) == 1 for
            row in axes(masks, 1)
        ) || error("Disabled trimming changed sequence start.")
    finally
        if previous_trim === nothing
            delete!(ENV, "JEFF_METAL_TRIM_PADDING")
        else
            ENV["JEFF_METAL_TRIM_PADDING"] = previous_trim
        end
    end
    println("Validated leading padding selection, interior masks, and disabled trimming.")
    probe = Metal.MtlArray(zeros(Float32, 7))
    extension.launch_cached_kernel!(kernel_handle_probe!, probe; threads = 32, groups = 1)
    all(==(1.0f0), Array(probe)) || error("Cached kernel probe mismatch.")
    types = Tuple{typeof(Metal.mtlconvert(probe))}
    first_kernel = extension.cached_mtlfunction(kernel_handle_probe!, types)
    GC.gc(true)
    extension.cached_mtlfunction(kernel_handle_probe!, types) === first_kernel ||
        error("Kernel handle was not reused after GC.")
    Core.eval(@__MODULE__, quote
        function kernel_handle_probe!(output)
            index = Int32(Metal.thread_position_in_grid_1d())
            if index <= length(output)
                @inbounds output[index] = 2.0f0
            end
            return
        end
    end)
    Base.invokelatest(
        extension.launch_cached_kernel!,
        kernel_handle_probe!,
        probe;
        threads = 32,
        groups = 1,
    )
    all(==(2.0f0), Array(probe)) || error("Cached kernel ignored a method update.")
    for value in (3.0f0, 4.0f0)
        captured = captured_kernel_handle_probe(value)
        extension.launch_cached_kernel!(captured, probe; threads = 32, groups = 1)
        GC.gc(true)
        all(==(value), Array(probe)) || error("Stateful kernel fallback lost its capture.")
    end
    println("Validated kernel handle reuse, GC, and method-update invalidation.")
    verify_metal_command_buffers()
    queue = Metal.global_queue(Metal.device())
    for width in (7, 128, 1024), sequence_length in (1, 9, 65)
        host = reshape(sin.(Float32.(1:(width*sequence_length))), width, sequence_length)
        mixed = cos.(host)
        weight = sin.(Float32.(1:width))
        expected_residual = host .+ mixed
        expected = QwenDecisionCore.native_rms(expected_residual, weight, 1.0f-6)
        gpu_host, gpu_mixed, gpu_weight = Metal.MtlArray.((host, mixed, weight))
        residual, normalized =
            extension.residual_input_rms!(gpu_host, gpu_mixed, gpu_weight, 1.0f-6)
        residual === gpu_host || error("Input RMS must reuse its residual.")
        Array(residual) == expected_residual || error("Input RMS residual mismatch.")
        isapprox(Array(normalized), expected; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("Input RMS normalization mismatch.")
        column = sequence_length:sequence_length
        expected_residual[:, column] .+= mixed[:, column]
        _, normalized = extension.residual_input_rms!(
            view(gpu_host, :, column),
            view(gpu_mixed, :, column),
            gpu_weight,
            1.0f-6,
        )
        GC.gc(true)
        Array(gpu_host) == expected_residual || error("View RMS changed other columns.")
        expected = QwenDecisionCore.native_rms(expected_residual[:, column], weight, 1.0f-6)
        isapprox(Array(normalized), expected; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("View input RMS mismatch after GC.")
    end
    println("Validated fused in-place residual/input RMS after GC.")
    for width in (7, 128, 1024), sequence_length in (0, 1, 9, 65)
        host = reshape(sin.(Float32.(1:(width*sequence_length))), width, sequence_length)
        mixed = cos.(host)
        gpu_host, gpu_mixed = Metal.MtlArray.((host, mixed))
        result = QwenDecisionCore.native_residual_add!(gpu_host, gpu_mixed)
        result === gpu_host || error("Residual addition must reuse its destination.")
        Array(result) == host .+ mixed || error("Dedicated residual addition mismatch.")
        GC.gc(true)
        QwenDecisionCore.native_residual_add!(gpu_host, gpu_host)
        Array(gpu_host) == 2.0f0 .* (host .+ mixed) ||
            error("Aliased residual addition mismatch.")
    end
    println("Validated in-place residual addition, including aliasing and empty inputs.")
    for width in (7, 128, 3584), sequence_length in (0, 1, 9, 65)
        gate = reshape(
            100.0f0 .* sin.(Float32.(1:(width*sequence_length))),
            width,
            sequence_length,
        )
        up = reshape(cos.(Float32.(1:(width*sequence_length))), size(gate))
        gpu_gate, gpu_up = Metal.MtlArray.((gate, up))
        result = QwenDecisionCore.native_mlp_gate!(gpu_gate, gpu_up)
        result === gpu_gate || error("MLP gate must reuse its destination.")
        isapprox(
            Array(result),
            QwenDecisionCore.native_silu.(gate) .* up;
            atol = 2.0f-5,
            rtol = 2.0f-5,
        ) || error("Dedicated MLP gate mismatch.")
        GC.gc(true)
    end
    println("Validated in-place MLP gate across widths, lengths, and extreme inputs.")
    for width in (7, 128, 3584), sequence_length in (1, 9)
        gate_weight = reshape(sin.(Float32.(1:(7width))), 7, width) .* 0.1f0
        up_weight = reshape(cos.(Float32.(1:(7width))), 7, width) .* 0.1f0
        down_weight = reshape(sin.(Float32.(1:(3width))), width, 3) .* 0.1f0
        host = reshape(cos.(Float32.(1:(7sequence_length))), 7, sequence_length)
        mlp = extension.PackedMLP(
            Metal.MtlArray(hcat(gate_weight, up_weight)),
            Metal.MtlArray(down_weight),
        )
        expected =
            transpose(down_weight) * (
                QwenDecisionCore.native_silu.(transpose(gate_weight) * host) .*
                (transpose(up_weight) * host)
            )
        gpu_host = Metal.MtlArray(host)
        for pass = 1:2
            actual = Array(QwenDecisionCore.native_mlp(mlp, gpu_host))
            isapprox(actual, expected; atol = 2.0f-4, rtol = 2.0f-4) ||
                error("Packed MLP mismatch.")
            GC.gc(true)
        end
    end
    println("Validated packed MLP projections against CPU arithmetic after GC.")
    for width in (7, 128, 1024), sequence_length in (1, 9, 65)
        host = reshape(sin.(Float32.(1:(width*sequence_length))), width, sequence_length)
        mask = Float32.(isodd.(1:sequence_length))
        actual =
            Array(extension.delta_masked_input(Metal.MtlArray(host), Metal.MtlArray(mask)))
        actual == host .* reshape(mask, 1, :) || error("Delta input mask mismatch.")
        GC.gc(true)
    end
    println("Validated delta input masking across widths and lengths.")
    for heads in (1, 3, 16), sequence_length in (1, 9, 65)
        a = reshape(
            5.0f0 .* sin.(Float32.(1:(heads*sequence_length))),
            heads,
            sequence_length,
        )
        b = reshape(
            5.0f0 .* cos.(Float32.(1:(heads*sequence_length))),
            heads,
            sequence_length,
        )
        a[1], b[1] = -100.0f0, -100.0f0
        a[end], b[end] = 100.0f0, 100.0f0
        dt_bias = collect(range(-5.0f0, 5.0f0; length = heads + 1))[1:heads]
        a_decay = -exp.(collect(range(-5.0f0, 5.0f0; length = heads + 1))[1:heads])
        beta, decay = extension.delta_gates(
            Metal.MtlArray(b),
            Metal.MtlArray(a),
            Metal.MtlArray(a_decay),
            Metal.MtlArray(dt_bias),
        )
        isapprox(
            Array(beta),
            QwenDecisionCore.native_sigmoid.(b);
            atol = 2.0f-6,
            rtol = 2.0f-5,
        ) || error("Fused delta beta mismatch.")
        isapprox(
            Array(decay),
            a_decay .* QwenDecisionCore.native_softplus.(a .+ dt_bias);
            atol = 2.0f-5,
            rtol = 2.0f-5,
        ) || error("Fused delta decay mismatch.")
        GC.gc(true)
    end
    println(
        "Validated fused delta beta/decay for head counts, lengths, and extreme inputs.",
    )
    previous_workspace_setting = get(ENV, "JEFF_METAL_WORKSPACE", nothing)
    previous_shape_setting = get(ENV, "JEFF_METAL_SHAPE_WORKSPACES", nothing)
    ENV["JEFF_METAL_WORKSPACE"] = "1"
    try
        host_weight = reshape(sin.(Float32.(1:35)), 7, 5)
        weight = Metal.MtlArray(host_weight)
        previous_array = nothing
        previous_values = nothing
        previous_feeds = nothing
        for sequence_length in (9, 9, 1, 1, 65, 65)
            input = reshape(cos.(Float32.(1:7sequence_length)), 7, sequence_length)
            device_input = Metal.MtlArray(input)
            actual = QwenDecisionCore.native_forward_scope(weight) do
                extension.launch_cached_kernel!(
                    captured_kernel_handle_probe(5.0f0),
                    probe;
                    threads = 32,
                    groups = 1,
                )
                QwenDecisionCore.native_host(
                    QwenDecisionCore.native_linear(weight, device_input),
                )
            end
            isapprox(
                actual,
                transpose(host_weight) * input;
                atol = 2.0f-5,
                rtol = 2.0f-5,
            ) || error("Workspace linear mismatch.")
            workspace = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
            workspace.command_queue === queue ||
                error("Workspace kernel launch uses a different task queue.")
            all(==(5.0f0), Array(probe)) ||
                error("Workspace kernel launch did not complete on readback.")
            array = only(workspace.slots)
            values = workspace.tensor_data[objectid(array)]
            feeds = workspace.feed_values[objectid(array)]
            all(value -> value === values[1], feeds) ||
                error("Completed feed retains input bindings.")
            length(workspace.feed_values) == 1 || error("Stale workspace feed values.")
            length(workspace.tensor_data) == 1 || error("Stale workspace tensor-data.")
            workspace.slot_indices == Dict(objectid(array) => 1) ||
                error("Stale workspace slot index.")
            if previous_array !== nothing && size(previous_array) == size(array)
                array === previous_array || error("Workspace array was not reused.")
                values === previous_values || error("Result value Vector was not reused.")
                feeds === previous_feeds || error("Feed value Vector was not reused.")
            end
            previous_array, previous_values = array, values
            previous_feeds = feeds
            GC.gc(true)
        end
        workspace = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
        previous_input_data = nothing
        previous_owned_input = nothing
        for sequence_length in (9, 9, 1, 1, 65, 65)
            actual = QwenDecisionCore.native_forward_scope(weight) do
                owned_input = extension.pooled_array(Float32, (7, sequence_length))
                fill!(owned_input, 1.0f0)
                result = QwenDecisionCore.native_linear(weight, owned_input)
                current_workspace = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
                data = current_workspace.tensor_data[objectid(owned_input)][1]
                if previous_owned_input !== nothing &&
                   size(previous_owned_input) == size(owned_input)
                    owned_input === previous_owned_input ||
                        error("Past slot array was not reused.")
                    data === previous_input_data ||
                        error("Past slot tensor-data was not reused.")
                end
                previous_owned_input, previous_input_data = owned_input, data
                QwenDecisionCore.native_host(result)
            end
            isapprox(
                actual,
                transpose(host_weight) * ones(Float32, 7, sequence_length);
                atol = 2.0f-5,
                rtol = 2.0f-5,
            ) || error("Past slot linear mismatch.")
            length(workspace.tensor_data) == 2 || error("Stale past slot tensor-data.")
            length(workspace.slot_indices) == 2 || error("Stale past slot index.")
            for (identifier, slot) in workspace.slot_indices
                objectid(workspace.slots[slot]) == identifier ||
                    error("Invalid workspace slot identity.")
            end
            GC.gc(true)
        end
        failed = try
            QwenDecisionCore.native_forward_scope(weight) do
                QwenDecisionCore.native_linear(weight, Metal.MtlArray(ones(Float32, 7, 3)))
                error("workspace failure probe")
            end
            false
        catch exception
            exception isa ErrorException && exception.msg == "workspace failure probe" || rethrow()
            true
        end
        failed && !workspace.active || error("Workspace exception cleanup failed.")
        length(workspace.slot_indices) == 1 ||
            error("Failed forward retains stale slot index.")
        extension.clear_forward_workspace!()
        isempty(workspace.slots) && isempty(workspace.tensor_data) ||
            error("Workspace retains slots after clear.")
        isempty(workspace.feed_values) || error("Workspace retains feeds after clear.")
        isempty(workspace.slot_indices) ||
            error("Workspace retains slot indices after clear.")
        !haskey(task_local_storage(), extension.FORWARD_WORKSPACE_KEY) ||
            error("Workspace remains task-local after clear.")
        gate_weight = reshape(sin.(Float32.(1:35)), 7, 5) .* 0.1f0
        up_weight = reshape(cos.(Float32.(1:35)), 7, 5) .* 0.1f0
        down_weight = reshape(sin.(Float32.(1:15)), 5, 3) .* 0.1f0
        packed_mlp = extension.PackedMLP(
            Metal.MtlArray(hcat(gate_weight, up_weight)),
            Metal.MtlArray(down_weight),
        )
        previous_slots = nothing
        previous_data = nothing
        for sequence_length in (9, 9, 1, 1, 65, 65)
            host = reshape(cos.(Float32.(1:(7sequence_length))), 7, sequence_length)
            input = Metal.MtlArray(host)
            expected =
                transpose(down_weight) * (
                    QwenDecisionCore.native_silu.(transpose(gate_weight) * host) .*
                    (transpose(up_weight) * host)
                )
            actual = QwenDecisionCore.native_forward_scope(input) do
                QwenDecisionCore.native_host(QwenDecisionCore.native_mlp(packed_mlp, input))
            end
            isapprox(actual, expected; atol = 2.0f-4, rtol = 2.0f-4) ||
                error("Packed workspace MLP mismatch.")
            workspace = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
            length(workspace.slots) ==
            length(workspace.tensor_data) ==
            length(workspace.slot_indices) ==
            3 || error("Packed workspace retains stale slots.")
            length(workspace.feed_values) == 2 ||
                error("Packed workspace retains stale feeds.")
            data = [workspace.tensor_data[objectid(slot)][1] for slot in workspace.slots]
            if previous_slots !== nothing &&
               size(first(previous_slots)) == size(first(workspace.slots))
                all(a === b for (a, b) in zip(previous_slots, workspace.slots)) ||
                    error("Packed slots not reused.")
                all(a === b for (a, b) in zip(previous_data, data)) ||
                    error("Packed tensor-data not reused.")
            end
            for (key, values) in workspace.feed_values
                all(value === workspace.tensor_data[key][1] for value in values) ||
                    error("Packed workspace retains input bindings.")
            end
            previous_slots, previous_data = copy(workspace.slots), data
            GC.gc(true)
        end
        try
            QwenDecisionCore.native_forward_scope(weight) do
                QwenDecisionCore.native_mlp(packed_mlp, Metal.MtlArray(ones(Float32, 7, 3)))
                error("packed workspace failure probe")
            end
        catch exception
            exception isa ErrorException &&
            exception.msg == "packed workspace failure probe" || rethrow()
        end
        !workspace.active || error("Packed workspace stays active after exception.")
        for (key, values) in workspace.feed_values
            all(value === workspace.tensor_data[key][1] for value in values) ||
                error("Failed packed workspace retains input bindings.")
        end
        extension.clear_forward_workspace!()
        isempty(workspace.slots) &&
        isempty(workspace.tensor_data) &&
        isempty(workspace.slot_indices) &&
        isempty(workspace.feed_values) ||
            error("Packed workspace retains objects after clear.")
        ENV["JEFF_METAL_SHAPE_WORKSPACES"] = "1"
        previous_shape_arrays = Dict{Int,Any}()
        previous_shape_data = Dict{Int,Any}()
        for sequence_length in (9, 1, 9, 1, 65)
            input = Metal.MtlArray(ones(Float32, 7, sequence_length))
            actual = QwenDecisionCore.native_forward_scope(weight, sequence_length) do
                QwenDecisionCore.native_host(QwenDecisionCore.native_linear(weight, input))
            end
            isapprox(
                actual,
                transpose(host_weight) * ones(Float32, 7, sequence_length);
                atol = 2.0f-5,
                rtol = 2.0f-5,
            ) || error("Shape workspace product mismatch.")
            bank = task_local_storage(extension.SHAPE_WORKSPACE_KEY)
            current = bank.entries[sequence_length]
            array = only(current.slots)
            data = current.tensor_data[objectid(array)][1]
            if haskey(previous_shape_arrays, sequence_length)
                array === previous_shape_arrays[sequence_length] ||
                    error("Shape workspace array not reused.")
                data === previous_shape_data[sequence_length] ||
                    error("Shape workspace tensor-data not reused.")
            end
            previous_shape_arrays[sequence_length], previous_shape_data[sequence_length] =
                array, data
            length(bank.entries) <= 2 || error("Shape workspace count limit exceeded.")
            current.bytes == array.maxsize || error("Workspace buffer accounting mismatch.")
            GC.gc(true)
        end
        bank = task_local_storage(extension.SHAPE_WORKSPACE_KEY)
        bank.order == [1, 65] || error("Shape workspace LRU mismatch.")
        bank.byte_limit = extension.BUFFER_PAGE
        QwenDecisionCore.native_forward_scope(weight, 9) do
            QwenDecisionCore.native_host(
                QwenDecisionCore.native_linear(weight, Metal.MtlArray(ones(Float32, 7, 9))),
            )
        end
        collect(keys(bank.entries)) == [9] ||
            error("Shape workspace byte limit not enforced.")
        bank.byte_limit = typemax(Int)
        QwenDecisionCore.native_forward_scope(weight, 9) do
            current = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
            QwenDecisionCore.native_forward_scope(weight, 99) do
                task_local_storage(extension.FORWARD_WORKSPACE_KEY) === current ||
                    error("Nested scope changed workspace.")
                QwenDecisionCore.native_linear(weight, Metal.MtlArray(ones(Float32, 7, 9)))
            end
            Metal.synchronize()
        end
        !haskey(bank.entries, 99) || error("Nested scope added shape workspace.")
        try
            QwenDecisionCore.native_forward_scope(weight, 1) do
                QwenDecisionCore.native_linear(weight, Metal.MtlArray(ones(Float32, 7, 1)))
                error("shape workspace failure probe")
            end
        catch exception
            exception isa ErrorException &&
            exception.msg == "shape workspace failure probe" || rethrow()
        end
        for entry in values(bank.entries)
            !entry.active || error("Shape workspace active after failure.")
            for (key, values) in entry.feed_values
                all(value === entry.tensor_data[key][1] for value in values) ||
                    error("Shape workspace retains input bindings.")
            end
        end
        entries = collect(values(bank.entries))
        extension.clear_forward_workspace!()
        isempty(bank.entries) && isempty(bank.order) ||
            error("Shape workspace bank not cleared.")
        all(
            isempty(entry.slots) &&
                isempty(entry.tensor_data) &&
                isempty(entry.feed_values) &&
                isempty(entry.slot_indices) &&
                entry.bytes == 0 for entry in entries
        ) || error("Shape workspace retains objects after clear.")
    finally
        if previous_workspace_setting === nothing
            delete!(ENV, "JEFF_METAL_WORKSPACE")
        else
            ENV["JEFF_METAL_WORKSPACE"] = previous_workspace_setting
        end
        if previous_shape_setting === nothing
            delete!(ENV, "JEFF_METAL_SHAPE_WORKSPACES")
        else
            ENV["JEFF_METAL_SHAPE_WORKSPACES"] = previous_shape_setting
        end
    end
    println("Validated workspace arrays/result Vectors across length changes and GC.")
    for width in (7, 128, 1024)
        host = reshape(sin.(Float32.(1:(width*11))), width, 11)
        embedding = Metal.MtlArray(host)
        for ids in (Int64[], Int64[0], Int64[10, 0, 5, 5, 10])
            actual = Array(QwenDecisionCore.native_gather(embedding, ids))
            actual == host[:, ids .+ 1] || error("Embedding gather mismatch.")
        end
        for ids in (Int64[-1], Int64[11], Int64[0, typemax(Int64)])
            caught = try
                QwenDecisionCore.native_gather(embedding, ids)
                false
            catch exception
                exception isa ArgumentError || rethrow()
                true
            end
            caught || error("Invalid gather ID was accepted.")
        end
        GC.gc(true)
    end
    println("Validated embedding gather widths, empty/repeated IDs, and invalid IDs.")
    owner_key = verify_weight_tensor_owner(extension)
    GC.gc(true)
    Metal.synchronize()
    GC.gc(true)
    !haskey(extension.WEIGHT_TENSOR_CACHE, owner_key) ||
        error("Dead weight owner remains cached.")
    println("Validated weight tensor reuse across lengths and owner GC cleanup.")
    maximum_error = 0.0f0
    for heads in (1, 3), transpose_a in ('N', 'T'), transpose_b in ('N', 'T')
        left_shape = transpose_a == 'N' ? (7, 5) : (5, 7)
        right_shape = transpose_b == 'N' ? (5, 9) : (9, 5)
        dims_a = heads == 1 ? left_shape : (left_shape..., heads)
        dims_b = heads == 1 ? right_shape : (right_shape..., heads)
        dims_c = heads == 1 ? (7, 9) : (7, 9, heads)
        left = reshape(sin.(Float32.(1:prod(dims_a))), dims_a)
        right = reshape(cos.(Float32.(1:prod(dims_b))), dims_b)
        a, b = Metal.MtlArray(left), Metal.MtlArray(right)
        output = extension.pooled_array(Float32, dims_c)
        expected = similar(left, dims_c)
        for head = 1:heads
            l, r = heads == 1 ? (left, right) : (left[:, :, head], right[:, :, head])
            l = transpose_a == 'N' ? l : transpose(l)
            r = transpose_b == 'N' ? r : transpose(r)
            if heads == 1
                expected .= l * r
            else
                expected[:, :, head] .= l * r
            end
        end
        for pass = 1:2
            # A product-only graph must overwrite poisoned destinations. Repeat
            # after GC/pool drainage to exercise cached Objective-C shape lifetimes.
            fill!(output, NaN32)
            extension.batched_matmul!(output, a, b, transpose_a, transpose_b)
            actual = Array(output)
            all(isfinite, actual) || error("Poisoned destination affected the product.")
            isapprox(actual, expected; atol = 2.0f-5, rtol = 2.0f-5) ||
                error("Product mismatch.")
            maximum_error = max(maximum_error, maximum(abs.(actual .- expected)))
            GC.gc(true)
        end
    end
    println(
        "Validated 8 matrix/batched product combinations × 2 passes with NaN destinations; max error ",
        maximum_error,
    )
    for width in (8, 33, 64, 128, 129, 256, 257, 512, 1024), centered in (true, false)
        println("Checking RMS width ", width, "; centered ", centered)
        flush(stdout)
        host = reshape(sin.(Float32.(1:(width*6))), width, 2, 3)
        weight = cos.(Float32.(1:width)) .* 0.1f0
        actual = Array(
            QwenDecisionCore.native_rms(
                Metal.MtlArray(host),
                Metal.MtlArray(weight),
                1.0f-6;
                centered,
            ),
        )
        expected = QwenDecisionCore.native_rms(host, weight, 1.0f-6; centered)
        isapprox(actual, expected; atol = 2.0f-5, rtol = 2.0f-5) || error("RMS mismatch.")
        mask = Float32[0, 1, 0, 1, 1, 1]
        gpu_mask = Metal.MtlArray(mask)
        masked = extension.fused_normalization(
            Metal.MtlArray(host),
            Metal.MtlArray(weight),
            1.0f-6,
            1.0f0,
            Val(true),
            centered ? Val(true) : Val(false),
            Val(true);
            mask = gpu_mask,
        )
        masked_expected = reshape(expected, width, 6) .* permutedims(mask)
        isapprox(
            reshape(Array(masked), width, 6),
            masked_expected;
            atol = 2.0f-5,
            rtol = 2.0f-5,
        ) || error("Masked RMS mismatch.")
        if centered
            matrix = reshape(host, width, 6)
            added = matrix .* 0.1f0
            residual, normalized = extension.residual_input_rms!(
                Metal.MtlArray(matrix),
                Metal.MtlArray(added),
                Metal.MtlArray(weight),
                1.0f-6,
                gpu_mask,
            )
            sum_expected = matrix .+ added
            norm_expected =
                QwenDecisionCore.native_rms(sum_expected, weight, 1.0f-6) .*
                permutedims(mask)
            isapprox(Array(residual), sum_expected; atol = 2.0f-5, rtol = 2.0f-5) ||
                error("Masked normalization modified the residual mask.")
            isapprox(Array(normalized), norm_expected; atol = 2.0f-5, rtol = 2.0f-5) ||
                error("Masked residual RMS mismatch.")
        end
        GC.gc(true)
    end
    println("Validated RMS normalization widths and centered/noncentered weights.")
    for (width, heads, kv_heads, rotary_dim, length, theta) in (
        (4, 2, 1, 4, 3, 10000.0f0),
        (7, 3, 3, 2, 1, 1000000.0f0),
        (256, 8, 2, 64, 65, 10000000.0f0),
        (256, 8, 2, 128, 65, 10000000.0f0),
        (256, 8, 2, 128, 65, 10000.0f0),
        (256, 8, 2, 0, 3, 10000.0f0),
    )
        cfg = (
            head_dim = width,
            heads = heads,
            kv_heads = kv_heads,
            rotary_dim = rotary_dim,
            rope_theta = theta,
            eps = 1.0f-6,
        )
        println("Checking fused heads: ", cfg, "; length ", length)
        flush(stdout)
        qgate = reshape(sin.(Float32.(1:(2width*heads*length))), 2width * heads, length)
        key = reshape(cos.(Float32.(1:(width*kv_heads*length))), width * kv_heads, length)
        value = sin.(key)
        weight = cos.(Float32.(1:width)) .* 0.1f0
        gpu_q, gpu_k, gpu_v, gpu_w = Metal.MtlArray.((qgate, key, value, weight))
        tables = extension.rope_tables(gpu_q, cfg, length)
        repeated = extension.rope_tables(gpu_q, cfg, length)
        tables[1] === repeated[1] && tables[2] === repeated[2] ||
            error("RoPE cache did not reuse its tables.")
        alternate = extension.rope_tables(gpu_q, cfg, length + 1)
        Metal.synchronize()
        GC.gc(true)
        restored = extension.rope_tables(gpu_q, cfg, length)
        tables[1] === restored[1] && tables[2] === restored[2] ||
            error("RoPE cache replaced tables after a length switch and GC.")
        alternate[1] !== restored[1] || error("RoPE cache reused the wrong length.")
        extension.rope_tables(gpu_q, cfg, length + 2)
        queue = objectid(Metal.global_queue(Metal.device()))
        entries = extension.ROPE_TABLE_CACHE[queue]
        Base.length(entries) == 2 || error("RoPE cache exceeded its entry limit.")
        any(table -> table.key == (rotary_dim, length + 1, theta), entries) &&
            error("RoPE cache did not evict the least recently used length.")
        actual_q = Array(extension.prepare_query(gpu_q, gpu_w, tables, cfg, length))
        gpu_key, gpu_value =
            extension.prepare_key_value(gpu_k, gpu_v, gpu_w, tables, cfg, length)
        actual_k, actual_v = Array(gpu_key), Array(gpu_value)
        qheads = reshape(qgate, 2width, heads, length)
        expected_q = permutedims(
            QwenDecisionCore.native_rope(
                QwenDecisionCore.native_rms(qheads[1:width, :, :], weight, cfg.eps),
                cfg,
            ),
            (1, 3, 2),
        )
        expected_k = QwenDecisionCore.native_rope(
            QwenDecisionCore.native_rms(
                reshape(key, width, kv_heads, length),
                weight,
                cfg.eps,
            ),
            cfg,
        )
        mapping = [cld(head, heads ÷ kv_heads) for head = 1:heads]
        expected_k = permutedims(expected_k[:, mapping, :], (1, 3, 2))
        expected_v =
            permutedims(reshape(value, width, kv_heads, length)[:, mapping, :], (1, 3, 2))
        isapprox(actual_q, expected_q; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("Fused query mismatch.")
        isapprox(actual_k, expected_k; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("Fused key mismatch.")
        actual_v == expected_v || error("Grouped value layout mismatch.")
        merged = Array(extension.merge_gate(gpu_value, gpu_q, cfg, length))
        expected_merged = reshape(
            permutedims(expected_v, (1, 3, 2)) .*
            QwenDecisionCore.native_sigmoid.(qheads[(width+1):end, :, :]),
            width * heads,
            length,
        )
        isapprox(merged, expected_merged; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("Merged gate mismatch.")
        GC.gc(true)
    end
    println("Validated fused RMS/RoPE, grouped head layouts, gates, and cache keys.")
    for width in (8, 33, 64, 128, 129, 256, 257, 512, 1024)
        host = reshape(sin.(Float32.(1:(width*6))), width, 2, 3)
        host[:, 1, 1] .= 0.0f0
        gate = reshape(12.0f0 .* cos.(Float32.(1:(width*6))), size(host))
        weight = cos.(Float32.(1:width)) .* 0.1f0
        expected =
            QwenDecisionCore.native_rms(host, weight, 1.0f-6; centered = false) .*
            QwenDecisionCore.native_silu.(gate)
        gpu_host, gpu_gate, gpu_weight = Metal.MtlArray.((host, gate, weight))
        for pass = 1:2
            actual = Array(extension.rms_silu_gate(gpu_host, gpu_gate, gpu_weight, 1.0f-6))
            flattened = Array(
                extension.rms_silu_gate(
                    gpu_host,
                    gpu_gate,
                    gpu_weight,
                    1.0f-6,
                    (2width, 3),
                ),
            )
            isapprox(
                flattened,
                reshape(expected, 2width, 3);
                atol = 2.0f-5,
                rtol = 2.0f-5,
            ) || error("Flattened RMS/SiLU gate mismatch.")
            isapprox(actual, expected; atol = 2.0f-5, rtol = 2.0f-5) ||
                error("RMS SiLU gate mismatch.")
            GC.gc(true)
        end
    end
    println("Validated fused RMS/SiLU gates at cached and fallback widths after GC.")
    for (width, heads, value_width) in ((7, 2, 3), (128, 2, 7), (256, 1, 9)),
        length in (1, 9, 65)

        value_heads = 2heads
        channels = 2width * heads + value_width * value_heads
        mixed = reshape(sin.(Float32.(1:(channels*length))), channels, length)
        mixed[:, 1] .= 0.0f0 # Zero norms exercise epsilon and layout boundaries.
        cfg = (key_dim = width, key_heads = heads)
        gpu_mixed = Metal.MtlArray(mixed)
        key_width = width * heads
        q = Array(extension.packed_qk(gpu_mixed, cfg, length, 0, sqrt(Float32(width))))
        k = Array(extension.packed_qk(gpu_mixed, cfg, length, key_width, 1.0f0))
        expected_q = reshape(mixed[1:key_width, :], width, heads, length)
        expected_k = reshape(mixed[(key_width+1):2key_width, :], width, heads, length)
        expected_q =
            expected_q ./
            (sqrt.(sum(abs2, expected_q; dims = 1) .+ 1.0f-6) .* sqrt(Float32(width)))
        expected_k = expected_k ./ sqrt.(sum(abs2, expected_k; dims = 1) .+ 1.0f-6)
        isapprox(q, expected_q; atol = 2.0f-6, rtol = 2.0f-5) || error("Packed Q mismatch.")
        isapprox(k, expected_k; atol = 2.0f-6, rtol = 2.0f-5) || error("Packed K mismatch.")
        paired_q, paired_k = extension.packed_qk_pair(gpu_mixed, cfg, length)
        isapprox(Array(paired_q), expected_q; atol = 2.0f-6, rtol = 2.0f-5) ||
            error("Paired packed Q mismatch.")
        isapprox(Array(paired_k), expected_k; atol = 2.0f-6, rtol = 2.0f-5) ||
            error("Paired packed K mismatch.")
        beta = fill(0.5f0, value_heads, length)
        decay = reshape(
            range(-12.0f0, 0.0f0; length = value_heads * length),
            value_heads,
            length,
        )
        gpu_q, gpu_k, gpu_beta, gpu_decay = Metal.MtlArray.((q, k, beta, decay))
        output = extension.pooled_array(Float32, (value_width, value_heads, length))
        Metal.@metal threads = (32, 8) groups = (cld(value_width, 8), value_heads) extension.delta_recurrent_kernel!(
            output,
            gpu_q,
            gpu_k,
            gpu_mixed,
            gpu_beta,
            gpu_decay,
            Int32(width),
            Int32(value_width),
            Int32(2key_width),
            Int32(2),
            Int32(length),
            Val(cld(width, 32)),
            Val(8),
        )
        expected = zeros(Float32, value_width, value_heads, length)
        for head = 1:value_heads
            key_head = cld(head, 2)
            state = zeros(Float32, width, value_width)
            for token = 1:length
                query, key = q[:, key_head, token], k[:, key_head, token]
                state .*= exp(decay[head, token])
                v = mixed[
                    (2key_width+(head-1)*value_width+1):(2key_width+head*value_width),
                    token,
                ]
                correction = (v - transpose(state) * key) .* beta[head, token]
                state .+= key * transpose(correction)
                expected[:, head, token] = transpose(state) * query
            end
        end
        isapprox(Array(output), expected; atol = 2.0f-5, rtol = 2.0f-4) ||
            error("Packed V recurrent mismatch.")
        gpu_factor = exp.(gpu_decay)
        Metal.@metal threads = (32, 8) groups = (cld(value_width, 8), value_heads) extension.delta_recurrent_kernel!(
            output,
            gpu_q,
            gpu_k,
            gpu_mixed,
            gpu_beta,
            gpu_factor,
            Int32(width),
            Int32(value_width),
            Int32(2key_width),
            Int32(2),
            Int32(length),
            Val(cld(width, 32)),
            Val(8),
            Val(true),
        )
        isapprox(Array(output), expected; atol = 2.0f-5, rtol = 2.0f-4) ||
            error("Precomputed decay-factor recurrent mismatch.")
        GC.gc(true)
    end
    println("Validated packed Q/K normalization and direct V recurrent reads.")
    verify_batched_causal_depthwise(extension)
    verify_batched_delta_recurrent(extension)
    verify_batched_masked_softmax(extension)
    verify_batched_full_attention(extension)
    verify_batched_forward_workspace(extension)
    verify_fused_delta_mask_workspace(extension)
end

main()
