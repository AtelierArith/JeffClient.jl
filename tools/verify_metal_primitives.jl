using JeffClient
using LinearAlgebra
import Metal

function main()
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    extension = Base.get_extension(JeffClient, :JeffClientMetalExt)
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
    for width in (8, 128, 256, 1024), centered in (true, false)
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
        beta = fill(0.5f0, value_heads, length)
        decay = fill(-0.1f0, value_heads, length)
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
        GC.gc(true)
    end
    println("Validated packed Q/K normalization and direct V recurrent reads.")
end

main()
