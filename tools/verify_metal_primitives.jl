using JeffClient
using LinearAlgebra
import Metal

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

function verify_weight_tensor_owner(extension)
    host_weight = reshape(sin.(Float32.(1:35)), 7, 5)
    weight = Metal.MtlArray(host_weight)
    key = objectid(weight)
    first_data = nothing
    for length in (1, 9, 65, 1)
        input = reshape(cos.(Float32.(1:7length)), 7, length)
        actual = Array(JeffClient.native_linear(weight, Metal.MtlArray(input)))
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
    Metal.synchronize()
    extension.clear_mps_command_cache!()
    queue = Metal.global_queue(Metal.device())
    owner = Metal.ensure_cmdbuf!(queue)
    command = extension.batched_mps_command_buffer(queue)
    extension.batched_mps_command_buffer(queue) === command ||
        error("MPS command wrapper was not reused.")
    pointer(command.commandBuffer) == pointer(owner) || error("MPS command owner mismatch.")
    GC.gc(true)
    extension.batched_mps_command_buffer(queue) === command ||
        error("MPS command wrapper lost after GC.")
    Metal.flush!(queue)
    next_command = extension.batched_mps_command_buffer(queue)
    next_command !== command || error("MPS command wrapper reused after submission.")
    pointer(next_command.commandBuffer) != pointer(owner) ||
        error("Submitted command buffer reused.")
    JeffClient.native_host(Metal.MtlArray(Float32[1]))
    cache = task_local_storage()[extension.MPS_COMMAND_CACHE_KEY]
    cache.owner === nothing && cache.command === nothing ||
        error("Completed MPS command retained.")
    println("Validated MPS command wrapper identity, submission boundaries, GC, and clear.")
    for width in (7, 128, 1024), sequence_length in (1, 9, 65)
        host = reshape(sin.(Float32.(1:(width*sequence_length))), width, sequence_length)
        mixed = cos.(host)
        weight = sin.(Float32.(1:width))
        expected_residual = host .+ mixed
        expected = JeffClient.native_rms(expected_residual, weight, 1.0f-6)
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
        expected = JeffClient.native_rms(expected_residual[:, column], weight, 1.0f-6)
        isapprox(Array(normalized), expected; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("View input RMS mismatch after GC.")
    end
    println("Validated fused in-place residual/input RMS after GC.")
    for width in (7, 128, 1024), sequence_length in (0, 1, 9, 65)
        host = reshape(sin.(Float32.(1:(width*sequence_length))), width, sequence_length)
        mixed = cos.(host)
        gpu_host, gpu_mixed = Metal.MtlArray.((host, mixed))
        result = JeffClient.native_residual_add!(gpu_host, gpu_mixed)
        result === gpu_host || error("Residual addition must reuse its destination.")
        Array(result) == host .+ mixed || error("Dedicated residual addition mismatch.")
        GC.gc(true)
        JeffClient.native_residual_add!(gpu_host, gpu_host)
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
        result = JeffClient.native_mlp_gate!(gpu_gate, gpu_up)
        result === gpu_gate || error("MLP gate must reuse its destination.")
        isapprox(
            Array(result),
            JeffClient.native_silu.(gate) .* up;
            atol = 2.0f-5,
            rtol = 2.0f-5,
        ) || error("Dedicated MLP gate mismatch.")
        GC.gc(true)
    end
    println("Validated in-place MLP gate across widths, lengths, and extreme inputs.")
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
            JeffClient.native_sigmoid.(b);
            atol = 2.0f-6,
            rtol = 2.0f-5,
        ) || error("Fused delta beta mismatch.")
        isapprox(
            Array(decay),
            a_decay .* JeffClient.native_softplus.(a .+ dt_bias);
            atol = 2.0f-5,
            rtol = 2.0f-5,
        ) || error("Fused delta decay mismatch.")
        GC.gc(true)
    end
    println(
        "Validated fused delta beta/decay for head counts, lengths, and extreme inputs.",
    )
    previous_workspace_setting = get(ENV, "JEFF_METAL_WORKSPACE", nothing)
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
            actual = JeffClient.native_forward_scope(weight) do
                JeffClient.native_host(JeffClient.native_linear(weight, device_input))
            end
            isapprox(
                actual,
                transpose(host_weight) * input;
                atol = 2.0f-5,
                rtol = 2.0f-5,
            ) || error("Workspace linear mismatch.")
            workspace = task_local_storage(extension.FORWARD_WORKSPACE_KEY)
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
            actual = JeffClient.native_forward_scope(weight) do
                owned_input = extension.pooled_array(Float32, (7, sequence_length))
                fill!(owned_input, 1.0f0)
                result = JeffClient.native_linear(weight, owned_input)
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
                JeffClient.native_host(result)
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
            JeffClient.native_forward_scope(weight) do
                JeffClient.native_linear(weight, Metal.MtlArray(ones(Float32, 7, 3)))
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
        cache = task_local_storage()[extension.MPS_COMMAND_CACHE_KEY]
        cache.owner === nothing && cache.command === nothing ||
            error("Failed forward retains MPS command.")
        extension.clear_forward_workspace!()
        isempty(workspace.slots) && isempty(workspace.tensor_data) ||
            error("Workspace retains slots after clear.")
        isempty(workspace.feed_values) || error("Workspace retains feeds after clear.")
        isempty(workspace.slot_indices) ||
            error("Workspace retains slot indices after clear.")
        !haskey(task_local_storage(), extension.FORWARD_WORKSPACE_KEY) ||
            error("Workspace remains task-local after clear.")
    finally
        if previous_workspace_setting === nothing
            delete!(ENV, "JEFF_METAL_WORKSPACE")
        else
            ENV["JEFF_METAL_WORKSPACE"] = previous_workspace_setting
        end
    end
    println("Validated workspace arrays/result Vectors across length changes and GC.")
    for width in (7, 128, 1024)
        host = reshape(sin.(Float32.(1:(width*11))), width, 11)
        embedding = Metal.MtlArray(host)
        for ids in (Int64[], Int64[0], Int64[10, 0, 5, 5, 10])
            actual = Array(JeffClient.native_gather(embedding, ids))
            actual == host[:, ids .+ 1] || error("Embedding gather mismatch.")
        end
        for ids in (Int64[-1], Int64[11], Int64[0, typemax(Int64)])
            caught = try
                JeffClient.native_gather(embedding, ids)
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
            JeffClient.native_rms(
                Metal.MtlArray(host),
                Metal.MtlArray(weight),
                1.0f-6;
                centered,
            ),
        )
        expected = JeffClient.native_rms(host, weight, 1.0f-6; centered)
        isapprox(actual, expected; atol = 2.0f-5, rtol = 2.0f-5) || error("RMS mismatch.")
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
        actual_q = Array(extension.prepare_query(gpu_q, gpu_w, tables, cfg, length))
        gpu_key, gpu_value =
            extension.prepare_key_value(gpu_k, gpu_v, gpu_w, tables, cfg, length)
        actual_k, actual_v = Array(gpu_key), Array(gpu_value)
        qheads = reshape(qgate, 2width, heads, length)
        expected_q = permutedims(
            JeffClient.native_rope(
                JeffClient.native_rms(qheads[1:width, :, :], weight, cfg.eps),
                cfg,
            ),
            (1, 3, 2),
        )
        expected_k = JeffClient.native_rope(
            JeffClient.native_rms(reshape(key, width, kv_heads, length), weight, cfg.eps),
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
            JeffClient.native_sigmoid.(qheads[(width+1):end, :, :]),
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
            JeffClient.native_rms(host, weight, 1.0f-6; centered = false) .*
            JeffClient.native_silu.(gate)
        gpu_host, gpu_gate, gpu_weight = Metal.MtlArray.((host, gate, weight))
        for pass = 1:2
            actual = Array(extension.rms_silu_gate(gpu_host, gpu_gate, gpu_weight, 1.0f-6))
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
        Metal.@metal threads=(32, 8) groups=(cld(value_width, 8), value_heads) extension.delta_recurrent_kernel!(
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
        Metal.@metal threads=(32, 8) groups=(cld(value_width, 8), value_heads) extension.delta_recurrent_kernel!(
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
end

main()
