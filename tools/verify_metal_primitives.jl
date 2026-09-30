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
        # A product-only graph must overwrite every destination element even
        # when recycled storage is poisoned; zero*NaN would still be NaN.
        fill!(output, NaN32)
        extension.batched_matmul!(output, a, b, transpose_a, transpose_b)
        actual = Array(output)
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
        all(isfinite, actual) || error("Poisoned destination affected the product.")
        isapprox(actual, expected; atol = 2.0f-5, rtol = 2.0f-5) ||
            error("Product mismatch.")
        maximum_error = max(maximum_error, maximum(abs.(actual .- expected)))
        GC.gc(true)
    end
    println(
        "Validated 8 matrix/batched products with NaN destinations; max error ",
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
end

main()
