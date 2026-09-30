# One logical 32-lane SIMD group normalizes a column. Fixed-size tuples keep
# inputs in registers through the reduction and output, without scratch arrays.
struct NormalizationConfig{PARTS,MEAN,CENTERED,WEIGHTED,RESIDUAL,GATED}
    eps::Float32
    factor::Float32
    width::Int32
    columns::Int32
end

function normalization_config(
    width,
    columns,
    eps,
    factor,
    ::Val{MEAN},
    ::Val{CENTERED},
    ::Val{WEIGHTED},
    ::Val{RESIDUAL},
    ::Val{GATED},
) where {MEAN,CENTERED,WEIGHTED,RESIDUAL,GATED}
    return NormalizationConfig{cld(width, 32),MEAN,CENTERED,WEIGHTED,RESIDUAL,GATED}(
        Float32(eps),
        Float32(factor),
        Int32(width),
        Int32(columns),
    )
end

function normalization_kernel!(
    output,
    input,
    added,
    residual,
    weight,
    gate,
    config::NormalizationConfig{PARTS,MEAN,CENTERED,WEIGHTED,RESIDUAL,GATED},
) where {PARTS,MEAN,CENTERED,WEIGHTED,RESIDUAL,GATED}
    eps, factor, width, columns = config.eps, config.factor, config.width, config.columns
    local_index = Metal.thread_position_in_threadgroup_2d()
    group = Metal.threadgroup_position_in_grid_2d().x
    lane = Int32(local_index.x)
    column = (Int32(group) - Int32(1)) * Int32(8) + Int32(local_index.y)
    if column <= columns
        offset = (column - Int32(1)) * width
        values = ntuple(Val(PARTS)) do part
            row = lane + Int32(32 * (part - 1))
            if row <= width
                value = @inbounds input[offset+row]
                if RESIDUAL
                    value += @inbounds added[offset+row]
                    @inbounds residual[offset+row] = value
                end
                value
            else
                0.0f0
            end
        end
        # Laya uses an explicit accumulator here. Base's tuple map/sum at
        # 32 elements crashed LLVM's Metal inliner on the real hidden width.
        squared = 0.0f0
        @inbounds for part = 1:PARTS
            squared += abs2(values[part])
        end
        squared = warp_sum(squared)
        variance = MEAN ? squared / Float32(width) : squared
        denominator = sqrt(variance + eps) * factor
        inverse = inv(denominator)
        @inbounds for part = 1:PARTS
            row = lane + Int32(32 * (part - 1))
            if row <= width
                normalized = MEAN ? values[part] * inverse : values[part] / denominator
                if WEIGHTED
                    w = CENTERED ? 1.0f0 + weight[row] : weight[row]
                    normalized *= w
                end
                if GATED
                    normalized *= JeffClient.native_silu(gate[offset+row])
                end
                output[offset+row] = normalized
            end
        end
    end
    return
end

function launch_normalization_parts!(
    output,
    input,
    added,
    residual,
    weight,
    gate,
    eps,
    factor,
    ::Val{MEAN},
    ::Val{CENTERED},
    ::Val{WEIGHTED},
    ::Val{RESIDUAL},
    ::Val{GATED},
    ::Val{PARTS},
) where {MEAN,CENTERED,WEIGHTED,RESIDUAL,GATED,PARTS}
    width = size(input, 1)
    columns = length(input) ÷ width
    config = NormalizationConfig{PARTS,MEAN,CENTERED,WEIGHTED,RESIDUAL,GATED}(
        Float32(eps),
        Float32(factor),
        Int32(width),
        Int32(columns),
    )
    launch_cached_kernel!(
        normalization_kernel!,
        output,
        input,
        added,
        residual,
        weight,
        gate,
        config;
        threads = (32, 8),
        groups = (cld(columns, 8), 1),
    )
    return nothing
end

function launch_normalization!(
    output,
    input,
    added,
    residual,
    weight,
    gate,
    eps,
    factor,
    mean,
    centered,
    weighted,
    residual_flag,
    gated,
)
    width = size(input, 1)
    if width == 1024
        return launch_normalization_parts!(
            output,
            input,
            added,
            residual,
            weight,
            gate,
            eps,
            factor,
            mean,
            centered,
            weighted,
            residual_flag,
            gated,
            Val(32),
        )
    elseif width == 128
        return launch_normalization_parts!(
            output,
            input,
            added,
            residual,
            weight,
            gate,
            eps,
            factor,
            mean,
            centered,
            weighted,
            residual_flag,
            gated,
            Val(4),
        )
    elseif width == 256
        return launch_normalization_parts!(
            output,
            input,
            added,
            residual,
            weight,
            gate,
            eps,
            factor,
            mean,
            centered,
            weighted,
            residual_flag,
            gated,
            Val(8),
        )
    elseif 0 < width <= 32
        return launch_normalization_parts!(
            output,
            input,
            added,
            residual,
            weight,
            gate,
            eps,
            factor,
            mean,
            centered,
            weighted,
            residual_flag,
            gated,
            Val(1),
        )
    end
    columns = length(input) ÷ width
    config = normalization_config(
        width,
        columns,
        eps,
        factor,
        mean,
        centered,
        weighted,
        residual_flag,
        gated,
    )
    Metal.@metal threads=(32, 8) groups=(cld(columns, 8), 1) normalization_kernel!(
        output,
        input,
        added,
        residual,
        weight,
        gate,
        config,
    )
    return nothing
end

function fused_normalization(x, weight, eps, factor, mean, centered, weighted)
    output = pooled_array(Float32, size(x))
    launch_normalization!(
        output,
        x,
        nothing,
        nothing,
        weight,
        nothing,
        eps,
        factor,
        mean,
        centered,
        weighted,
        Val(false),
        Val(false),
    )
    return output
end

function JeffClient.native_residual_rms(
    x::Metal.MtlMatrix{Float32},
    mixed::Metal.MtlMatrix{Float32},
    weight,
    eps,
)
    size(x) == size(mixed) || throw(DimensionMismatch("Residual shapes must match."))
    length(weight) == size(x, 1) ||
        throw(DimensionMismatch("RMS weight width must match the input."))
    size(x, 1) > 4096 && return invoke(
        JeffClient.native_residual_rms,
        Tuple{Any,Any,Any,Any},
        x,
        mixed,
        weight,
        eps,
    )
    residual = pooled_array(Float32, size(x))
    output = pooled_array(Float32, size(x))
    launch_normalization!(
        output,
        x,
        mixed,
        residual,
        weight,
        nothing,
        eps,
        1.0f0,
        Val(true),
        Val(true),
        Val(true),
        Val(true),
        Val(false),
    )
    return residual, output
end

function rms_silu_gate(x::Metal.MtlArray{Float32}, gate, weight, eps, output_dims = size(x))
    size(x) == size(gate) || throw(DimensionMismatch("RMS gate shapes must match."))
    length(weight) == size(x, 1) ||
        throw(DimensionMismatch("RMS weight width must match the input."))
    prod(output_dims) == length(x) ||
        throw(DimensionMismatch("RMS output must match input length."))
    if size(x, 1) > 4096
        output =
            JeffClient.native_rms(x, weight, eps; centered = false) .*
            JeffClient.native_silu.(gate)
        return reshape(output, output_dims)
    end
    output = pooled_array(Float32, output_dims)
    launch_normalization!(
        output,
        x,
        nothing,
        nothing,
        weight,
        gate,
        eps,
        1.0f0,
        Val(true),
        Val(false),
        Val(true),
        Val(false),
        Val(true),
    )
    return output
end

# The previous layer's MLP has already consumed its normalized residual.
# Read and update each residual element on the same queue while normalizing it.
function residual_input_rms!(residual, mixed, weight, eps)
    size(residual) == size(mixed) || throw(DimensionMismatch("Residual shapes must match."))
    length(weight) == size(residual, 1) ||
        throw(DimensionMismatch("RMS weight width must match."))
    output = pooled_array(Float32, size(residual))
    launch_normalization!(
        output,
        residual,
        mixed,
        residual,
        weight,
        nothing,
        eps,
        1.0f0,
        Val(true),
        Val(true),
        Val(true),
        Val(true),
        Val(false),
    )
    return residual, output
end

function JeffClient.native_hidden_forward(
    hidden::Metal.MtlMatrix{Float32},
    layers,
    mask,
    final_norm,
    cfg,
)
    size(hidden, 1) > 4096 && return invoke(
        JeffClient.native_hidden_forward,
        Tuple{Any,Any,Any,Any,Any},
        hidden,
        layers,
        mask,
        final_norm,
        cfg,
    )
    isempty(layers) && return JeffClient.native_rms(hidden[:, end:end], final_norm, cfg.eps)
    first_layer = first(layers)
    normalized = JeffClient.native_rms(hidden, first_layer.input_norm, cfg.eps)
    residual, mlp =
        JeffClient.native_layer_outputs(first_layer, hidden, normalized, mask, cfg)
    for index = 2:length(layers)
        layer = layers[index]
        hidden, normalized = residual_input_rms!(residual, mlp, layer.input_norm, cfg.eps)
        residual, mlp =
            JeffClient.native_layer_outputs(layer, hidden, normalized, mask, cfg)
    end
    # The last layer needs only the readout column, not a full residual sum.
    last_column = size(residual, 2)
    _, normalized = residual_input_rms!(
        view(residual, :, last_column:last_column),
        view(mlp, :, last_column:last_column),
        final_norm,
        cfg.eps,
    )
    return normalized
end

function JeffClient.native_rms(x::Metal.MtlArray{Float32}, weight, eps; centered = true)
    length(weight) == size(x, 1) ||
        throw(DimensionMismatch("RMS weight width must match the input."))
    size(x, 1) > 4096 &&
        return invoke(JeffClient.native_rms, Tuple{Any,Any,Any}, x, weight, eps; centered)
    # Split the Bool explicitly: Val(centered) alone leaves a runtime Val type
    # at this keyword boundary and caused JET dispatch reports in the forward.
    if centered
        return fused_normalization(x, weight, eps, 1.0f0, Val(true), Val(true), Val(true))
    else
        return fused_normalization(x, weight, eps, 1.0f0, Val(true), Val(false), Val(true))
    end
end

function mlp_gate_kernel!(gate, up)
    elements = Int32(length(gate))
    index = Int32(Metal.thread_position_in_grid_1d())
    if index <= elements
        @inbounds gate[index] = JeffClient.native_silu(gate[index]) * up[index]
    end
    return
end

function JeffClient.native_mlp_gate!(
    gate::Metal.MtlMatrix{Float32},
    up::Metal.MtlMatrix{Float32},
)
    size(gate) == size(up) || throw(DimensionMismatch("MLP gate shapes must match."))
    elements = length(gate)
    if elements > 0
        launch_cached_kernel!(
            mlp_gate_kernel!,
            gate,
            up;
            threads = 256,
            groups = cld(elements, 256),
        )
    end
    return gate
end

function residual_add_kernel!(residual, mixed)
    elements = Int32(length(residual))
    index = Int32(Metal.thread_position_in_grid_1d())
    if index <= elements
        @inbounds residual[index] += mixed[index]
    end
    return
end

function JeffClient.native_residual_add!(
    residual::Metal.MtlMatrix{Float32},
    mixed::Metal.MtlMatrix{Float32},
)
    size(residual) == size(mixed) || throw(DimensionMismatch("Residual shapes must match."))
    elements = length(residual)
    if elements > 0
        Metal.@metal threads=256 groups=cld(elements, 256) residual_add_kernel!(
            residual,
            mixed,
        )
    end
    return residual
end

function l2_normalize(x::Metal.MtlArray{Float32}, factor)
    size(x, 1) > 4096 && return x ./ (sqrt.(sum(abs2, x; dims = 1) .+ 1.0f-6) .* factor)
    # The unweighted specialization never reads this dummy weight argument.
    return fused_normalization(x, x, 1.0f-6, factor, Val(false), Val(false), Val(false))
end
