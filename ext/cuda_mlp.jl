struct PackedMLP{M}
    gate_up::M
    down::M
end

function JeffClient.native_attention_weights(::Val{:cuda}, attention)
    attention.kind == :delta || return attention
    packed = hcat(attention.qkv, attention.z, attention.a, attention.b)
    q = size(attention.qkv, 2)
    z = size(attention.z, 2)
    a = size(attention.a, 2)
    return merge(
        attention,
        (
            qkv = view(packed, :, 1:q),
            z = view(packed, :, (q+1):(q+z)),
            a = view(packed, :, (q+z+1):(q+z+a)),
            b = view(packed, :, (q+z+a+1):size(packed, 2)),
            packed_projection = packed,
        ),
    )
end

function JeffClient.native_mlp_weights(::Val{:cuda}, gate, up, down)
    return PackedMLP(hcat(gate, up), down)
end

function mlp_gate_kernel!(output, packed, width, tokens)
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(
            CUDA.threadIdx().x,
        )
    if index <= width*tokens
        row = rem(index-Int32(1), width)+Int32(1)
        token = (index-Int32(1))÷width+Int32(1)
        @inbounds output[index] =
            JeffClient.native_silu(packed[row, token])*packed[width+row, token]
    end
    return
end

function JeffClient.native_mlp(mlp::PackedMLP, x::CUDA.CuMatrix{Float32})
    projected = JeffClient.native_linear(mlp.gate_up, x)
    width = size(projected, 1)÷2
    gated = scratch(Float32, (width, size(x, 2)))
    CUDA.@cuda threads=256 blocks=cld(length(gated), 256) mlp_gate_kernel!(
        gated,
        projected,
        Int32(width),
        Int32(size(x, 2)),
    )
    return JeffClient.native_linear(mlp.down, gated)
end
