# CPU paths use column-major loops and preserve the generic GPU dispatch.
function causal_depthwise(input::Matrix{Float32}, weight::AbstractMatrix{Float32})
    kernel = size(weight, 1)
    channels, sequence_length = size(input)
    size(weight, 2) == channels ||
        throw(DimensionMismatch("Convolution channel counts differ."))
    output = zeros(Float32, size(input))
    for index = 1:kernel
        lag = kernel - index
        for token = (lag+1):sequence_length
            @inbounds for channel = 1:channels
                output[channel, token] += input[channel, token-lag] * weight[index, channel]
            end
        end
    end
    output .= native_silu.(output)
    return output
end

function native_hidden_forward(hidden::Matrix{Float32}, layers, mask, final_norm, cfg)
    isempty(layers) && return native_rms(hidden[:, end:end], final_norm, cfg.eps)
    for index = 1:(length(layers)-1)
        hidden = native_layer(layers[index], hidden, mask, cfg)
    end
    # Attention consumes the full context. The final MLP is position-wise,
    # and only the last position contributes to Jeff's trained readout.
    layer = last(layers)
    normalized = native_rms(hidden, layer.input_norm, cfg.eps)
    mixed =
        layer.attention.kind == :full ?
        full_attention(layer.attention, normalized, mask, cfg) :
        delta_attention(layer.attention, normalized, mask, cfg)
    residual = @views hidden[:, end:end] .+ mixed[:, end:end]
    normalized = native_rms(residual, layer.post_norm, cfg.eps)
    native_residual_add!(residual, native_mlp(layer.mlp, normalized))
    return native_rms(residual, final_norm, cfg.eps)
end
