struct NativeLayer{A,M,V}
    attention::A
    mlp::M
    input_norm::V
    post_norm::V
end

"""
    NativeBackend(checkpoint; device=:cpu)

Load Jeff's text-only Qwen3.5 model and readout directly from safetensors.
The complete forward pass is implemented in Julia. Inputs are prepared token
IDs and masks; tokenization is not part of this API. Only float32 inference,
default partial RoPE, and bias-free projections are currently supported.

Use `device=:metal` after importing Metal for Apple GPU execution.
CPU inference automatically selects its platform policy and skips leading
zero-mask positions without removing active tokens or interior mask holes.
"""
struct NativeBackend{E,R,L,N,C} <: AbstractDecisionBackend
    embedding::E
    readout::R
    layers::L
    final_norm::N
    config::C
    temperature::Float64
    max_options::Int
end

native_array(::Val{:cpu}, x) = x
native_array(device, x) = throw(
    ArgumentError(
        "Unsupported native device $device; import the required GPU package first.",
    ),
)
on_native_device(::Array, x::AbstractArray) = x
on_native_device(reference::AbstractArray, x::AbstractArray) =
    copyto!(similar(reference, eltype(x), size(x)), x)
native_host(x::Array) = x
native_host(x::AbstractArray) = Array(x)
native_forward_scope(f, reference) = f()
native_forward_scope(f, reference, sequence_length) = native_forward_scope(f, reference)
function native_sequence_start(reference, mask, row)
    start = first(axes(mask, 2))
    cpu_setting(:trim_padding) || return start
    while start < last(axes(mask, 2)) && mask[row, start] == 0
        start += 1
    end
    return start
end
native_batch_logits(backend, ids, mask) = nothing
native_mlp_weights(device, gate, up, down) = (; gate, up, down)
native_conv_weights(device, weight) = weight
native_gather(embedding, ids) = embedding[:, on_native_device(embedding, Int32.(ids .+ 1))]
native_gather(embedding::Matrix, ids) = embedding[:, ids .+ 1]

function NativeBackend(checkpoint::AbstractString; device::Symbol = :cpu)
    directory = resolve_checkpoint(checkpoint)
    source = JSON.parsefile(joinpath(directory, "config.json"))
    source["model_type"] == "qwen3_5" ||
        throw(ArgumentError("NativeBackend currently supports Qwen3.5 only."))
    decision = JSON.parsefile(joinpath(directory, "decision_config.json"))
    decision["format_version"] == 1 ||
        throw(ArgumentError("Unsupported decision checkpoint format."))
    cfg = source["text_config"]
    !cfg["attention_bias"] || throw(ArgumentError("Attention biases are not supported."))
    cfg["hidden_act"] == "silu" || throw(ArgumentError("Expected SiLU activation."))
    rope = cfg["rope_parameters"]
    rope["rope_type"] == "default" ||
        throw(ArgumentError("Only default RoPE is supported."))
    config = (
        hidden = Int(cfg["hidden_size"]),
        head_dim = Int(cfg["head_dim"]),
        heads = Int(cfg["num_attention_heads"]),
        kv_heads = Int(cfg["num_key_value_heads"]),
        key_heads = Int(cfg["linear_num_key_heads"]),
        value_heads = Int(cfg["linear_num_value_heads"]),
        key_dim = Int(cfg["linear_key_head_dim"]),
        value_dim = Int(cfg["linear_value_head_dim"]),
        eps = Float32(cfg["rms_norm_eps"]),
        rope_theta = Float32(rope["rope_theta"]),
        rotary_dim = Int(cfg["head_dim"] * rope["partial_rotary_factor"]),
    )
    config.value_heads % config.key_heads == 0 ||
        throw(ArgumentError("Value heads must be a multiple of key heads."))
    config.heads % config.kv_heads == 0 ||
        throw(ArgumentError("Query heads must be a multiple of KV heads."))
    files =
        isfile(joinpath(directory, "model.safetensors")) ? ["model.safetensors"] :
        unique(
            String.(
                collect(
                    values(
                        JSON.parsefile(
                            joinpath(directory, "model.safetensors.index.json"),
                        )["weight_map"],
                    ),
                ),
            ),
        )
    weights = Dict{String,Any}()
    for file in files
        safe_relative_path(file)
        merge!(
            weights,
            read_native_weights(
                joinpath(directory, file);
                select = name -> startswith(name, "language_model."),
                convert_array = x -> native_array(Val(device), x),
            ),
        )
    end
    take(name) = pop!(weights, "language_model." * name)
    layers = map(enumerate(cfg["layer_types"])) do (index, kind)
        prefix = "layers.$(index - 1)."
        get(suffix) = take(prefix * suffix)
        attention = if kind == "full_attention"
            (
                kind = :full,
                q = get("self_attn.q_proj.weight"),
                k = get("self_attn.k_proj.weight"),
                v = get("self_attn.v_proj.weight"),
                out = get("self_attn.o_proj.weight"),
                q_norm = get("self_attn.q_norm.weight"),
                k_norm = get("self_attn.k_norm.weight"),
            )
        elseif kind == "linear_attention"
            (
                kind = :delta,
                qkv = get("linear_attn.in_proj_qkv.weight"),
                z = get("linear_attn.in_proj_z.weight"),
                a = get("linear_attn.in_proj_a.weight"),
                b = get("linear_attn.in_proj_b.weight"),
                conv = native_conv_weights(
                    Val(device),
                    dropdims(get("linear_attn.conv1d.weight"); dims = 2),
                ),
                # Fixed inference weights: compute the decay multiplier once
                # on this backend, preserving its Float32 exp implementation.
                a_decay = -1.0f0 .* exp.(get("linear_attn.A_log")),
                dt_bias = get("linear_attn.dt_bias"),
                norm = get("linear_attn.norm.weight"),
                out = get("linear_attn.out_proj.weight"),
            )
        else
            throw(ArgumentError("Unsupported layer type $kind."))
        end
        mlp = native_mlp_weights(
            Val(device),
            get("mlp.gate_proj.weight"),
            get("mlp.up_proj.weight"),
            get("mlp.down_proj.weight"),
        )
        NativeLayer(
            attention,
            mlp,
            get("input_layernorm.weight"),
            get("post_attention_layernorm.weight"),
        )
    end
    # `map` joins the full-attention and DeltaNet layers at an abstract `A`,
    # which makes the forward loop's hidden state `Any`. Retain the two concrete
    # layer types as a small union so inference can split the dispatch.
    layer_type = Union{unique(typeof.(layers))...}
    layers = Vector{layer_type}(layers)
    embedding = take("embed_tokens.weight")
    final_norm = take("norm.weight")
    readout = read_native_weights(
        joinpath(directory, "readout.safetensors");
        convert_array = x -> native_array(Val(device), x),
    )["weight"]
    temperature = Float64(decision["temperature"])
    isfinite(temperature) && temperature > 0 ||
        throw(ArgumentError("Invalid checkpoint temperature."))
    limit = Int(decision["max_options"])
    1 <= limit <= size(readout, 2) <= 255 ||
        throw(ArgumentError("Invalid trained option limit or readout."))
    return NativeBackend(embedding, readout, layers, final_norm, config, temperature, limit)
end

native_matmul(a, b) = a * b
native_linear(weight, x) = native_matmul(transpose(weight), x)
native_sigmoid(x) = inv(one(x) + exp(-x))
native_silu(x) = x * native_sigmoid(x)
native_softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

function native_rms(x, weight, eps; centered = true)
    scale = inv.(sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ eps))
    w = reshape(weight, size(weight, 1), ntuple(_ -> 1, ndims(x) - 1)...)
    return x .* scale .* (centered ? (1.0f0 .+ w) : w)
end

function native_rope(x, cfg)
    half = cfg.rotary_dim ÷ 2
    freq = inv.(
        cfg.rope_theta .^ (Float32.(0:2:(cfg.rotary_dim-1)) ./ Float32(cfg.rotary_dim)),
    )
    theta = freq .* permutedims(Float32.(0:(size(x, 3)-1)))
    cosines = on_native_device(x, reshape(cos.(theta), half, 1, size(x, 3)))
    sines = on_native_device(x, reshape(sin.(theta), half, 1, size(x, 3)))
    first_half = x[1:half, :, :]
    second_half = x[(half+1):2half, :, :]
    return cat(
        first_half .* cosines .- second_half .* sines,
        second_half .* cosines .+ first_half .* sines,
        x[(2half+1):end, :, :];
        dims = 1,
    )
end

function full_attention(attention, x, mask, cfg)
    length = size(x, 2)
    qgate = reshape(native_linear(attention.q, x), 2cfg.head_dim, cfg.heads, length)
    q = native_rope(native_rms(qgate[1:cfg.head_dim, :, :], attention.q_norm, cfg.eps), cfg)
    k = native_rope(
        native_rms(
            reshape(native_linear(attention.k, x), cfg.head_dim, cfg.kv_heads, length),
            attention.k_norm,
            cfg.eps,
        ),
        cfg,
    )
    v = reshape(native_linear(attention.v, x), cfg.head_dim, cfg.kv_heads, length)
    scores_mask = on_native_device(
        x,
        Float32[
            i <= j && mask[i] == 1 ? 0 : -floatmax(Float32) for i = 1:length, j = 1:length
        ],
    )
    out = similar(x, cfg.head_dim * cfg.heads, length)
    native_full_heads!(out, q, k, v, qgate, scores_mask, cfg)
    return native_linear(attention.out, out)
end

function native_full_heads!(out, q, k, v, qgate, scores_mask, cfg)
    groups = cfg.heads ÷ cfg.kv_heads
    for head = 1:cfg.heads
        kv_head = cld(head, groups)
        scores =
            native_matmul(transpose(k[:, kv_head, :]), q[:, head, :]) ./
            sqrt(Float32(cfg.head_dim)) .+ scores_mask
        probabilities = exp.(scores .- maximum(scores; dims = 1))
        probabilities ./= sum(probabilities; dims = 1)
        values = native_matmul(v[:, kv_head, :], probabilities)
        gate = native_sigmoid.(qgate[(cfg.head_dim+1):end, head, :])
        out[((head-1)*cfg.head_dim+1):(head*cfg.head_dim), :] .= values .* gate
    end
    return out
end

function causal_depthwise(input, weight)
    kernel = size(weight, 1)
    length = size(input, 2)
    output = similar(input)
    fill!(output, 0.0f0)
    for index = 1:kernel
        lag = kernel - index
        lag >= length && continue
        output[:, (lag+1):length] .+=
            input[:, 1:(length-lag)] .* reshape(weight[index, :], :, 1)
    end
    return native_silu.(output)
end

# Forward substitution via BLAS on CPU. GPU backends can specialize this solve.
delta_solve(system::Matrix, rhs::Matrix) = UnitLowerTriangular(system) \ rhs

delta_solve(system, rhs) =
    throw(ArgumentError("A stable triangular solver is required for this array backend."))

function delta_attention(attention, x, mask, cfg)
    length = size(x, 2)
    masked = x .* on_native_device(x, reshape(Float32.(mask), 1, :))
    mixed = causal_depthwise(native_linear(attention.qkv, masked), attention.conv)
    key_width = cfg.key_dim * cfg.key_heads
    q = reshape(mixed[1:key_width, :], cfg.key_dim, cfg.key_heads, length)
    k = reshape(mixed[(key_width+1):2key_width, :], cfg.key_dim, cfg.key_heads, length)
    v = reshape(mixed[(2key_width+1):end, :], cfg.value_dim, cfg.value_heads, length)
    z = reshape(native_linear(attention.z, masked), cfg.value_dim, cfg.value_heads, length)
    beta = native_sigmoid.(native_linear(attention.b, masked))
    decay =
        attention.a_decay .*
        native_softplus.(native_linear(attention.a, masked) .+ attention.dt_bias)
    out = similar(x, cfg.value_dim * cfg.value_heads, length)
    groups = cfg.value_heads ÷ cfg.key_heads
    # All heads use the same causal masks. Construct/upload each chunk size
    # once instead of synchronizing a host-to-device copy in every head/chunk.
    full_size = min(64, length)
    full_lower = on_native_device(x, [i >= j for i = 1:full_size, j = 1:full_size])
    tail_size = mod(length, 64)
    tail_lower = if tail_size == 0 || tail_size == full_size
        full_lower
    else
        on_native_device(x, [i >= j for i = 1:tail_size, j = 1:tail_size])
    end
    for head = 1:cfg.value_heads
        kh = cld(head, groups)
        query = q[:, kh, :]
        key = k[:, kh, :]
        query ./= sqrt.(sum(abs2, query; dims = 1) .+ 1.0f-6) .* sqrt(Float32(cfg.key_dim))
        key ./= sqrt.(sum(abs2, key; dims = 1) .+ 1.0f-6)
        state = similar(x, cfg.value_dim, cfg.key_dim)
        fill!(state, 0.0f0)
        for start = 1:64:length
            span = start:min(start+63, length)
            n = Base.length(span)
            qc, kc, vc = query[:, span], key[:, span], v[:, head, span]
            bc = beta[head:head, span]
            cumulative = cumsum(decay[head:head, span]; dims = 2)
            lower = n == full_size ? full_lower : tail_lower
            differences = transpose(cumulative) .- cumulative
            pair_decay = exp.(ifelse.(lower, differences, -Inf32))
            system = native_matmul(transpose(kc .* bc), kc) .* pair_decay
            exp_decay = exp.(cumulative)
            new_values = transpose(delta_solve(system, copy(transpose(vc .* bc))))
            reading_keys =
                transpose(delta_solve(system, copy(transpose(kc .* bc .* exp_decay))))
            corrections = new_values .- native_matmul(state, reading_keys)
            intra = native_matmul(transpose(kc), qc) .* transpose(pair_decay)
            result =
                native_matmul(state, qc .* exp_decay) .+ native_matmul(corrections, intra)
            final_decay = exp.(cumulative[:, end:end])
            ending_keys = kc .* exp.(cumulative[:, end:end] .- cumulative)
            state =
                state .* final_decay .+ native_matmul(corrections, transpose(ending_keys))
            normalized = native_rms(result, attention.norm, cfg.eps; centered = false)
            out[((head-1)*cfg.value_dim+1):(head*cfg.value_dim), span] .=
                normalized .* native_silu.(z[:, head, span])
        end
    end
    return native_linear(attention.out, out)
end

native_residual_rms(x, mixed, weight, eps) = begin
    residual = x .+ mixed
    residual, native_rms(residual, weight, eps)
end

function native_mlp_gate!(gate, up)
    gate .= native_silu.(gate) .* up
    return gate
end
native_prepare_mask(reference, mask) = mask

function native_residual_add!(residual, mixed)
    residual .+= mixed
    return residual
end

function native_layer(layer, x, mask, cfg)
    normalized = native_rms(x, layer.input_norm, cfg.eps)
    residual, mlp = native_layer_outputs(layer, x, normalized, mask, cfg)
    return native_residual_add!(residual, mlp)
end

native_delta_attention(attention, x, mask, cfg, ::Val) =
    delta_attention(attention, x, mask, cfg)

function native_layer_outputs(layer, x, normalized, mask, cfg, premasked = Val(false))
    mixed =
        layer.attention.kind == :full ?
        full_attention(layer.attention, normalized, mask, cfg) :
        native_delta_attention(layer.attention, normalized, mask, cfg, premasked)
    residual, normalized = native_residual_rms(x, mixed, layer.post_norm, cfg.eps)
    mlp = native_mlp(layer.mlp, normalized)
    # residual is owned by this layer; its normalization has already consumed
    # the old values, so reuse it for the final sum on the same device queue.
    return residual, mlp
end

function native_mlp(mlp, x)
    gate = native_linear(mlp.gate, x)
    up = native_linear(mlp.up, x)
    return native_linear(mlp.down, native_mlp_gate!(gate, up))
end

function native_hidden_forward(hidden, layers, mask, final_norm, cfg)
    for layer in layers
        hidden = native_layer(layer, hidden, mask, cfg)
    end
    # RMS normalizes each column independently; readout needs only the last.
    return native_rms(hidden[:, end:end], final_norm, cfg.eps)
end

"""
    logits(backend::NativeBackend, inputs)

Compute uncalibrated `(batch, options)` scores from `input_ids` and
`attention_mask`, each with logical `(batch, sequence)` dimensions. Batches
are processed one sequence at a time by default; the experimental Metal
`JEFF_METAL_BATCHED=1` path computes samples together. No generation or KV cache is used.
"""
function logits(backend::NativeBackend, inputs::AbstractDict)
    Set(keys(inputs)) == Set(["input_ids", "attention_mask"]) ||
        throw(ArgumentError("Provide input_ids and attention_mask."))
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    ids isa AbstractMatrix{<:Integer} ||
        throw(ArgumentError("input_ids must be an integer matrix."))
    size(ids) == size(mask) ||
        throw(ArgumentError("Input IDs and mask must have identical shapes."))
    !isempty(ids) || throw(ArgumentError("Inputs must not be empty."))
    all(id -> 0 <= id < size(backend.embedding, 2), ids) ||
        throw(ArgumentError("Token ID is outside the vocabulary."))
    all(value -> value in (0, 1), mask) ||
        throw(ArgumentError("Mask values must be zero or one."))
    all(mask[:, end] .== 1) ||
        throw(ArgumentError("The final position must be active; use left padding."))
    batched = native_batch_logits(backend, ids, mask)
    batched === nothing || return batched
    result = Matrix{Float32}(undef, size(ids, 1), size(backend.readout, 2))
    for row in axes(ids, 1)
        first_token = native_sequence_start(backend.embedding, mask, row)
        scores = native_forward_scope(backend.embedding, size(ids, 2) - first_token + 1) do
            hidden = native_gather(backend.embedding, vec(ids[row, first_token:end]))
            row_mask = native_prepare_mask(hidden, vec(mask[row, first_token:end]))
            hidden = native_hidden_forward(
                hidden,
                backend.layers,
                row_mask,
                backend.final_norm,
                backend.config,
            )
            native_host(native_linear(backend.readout, hidden))
        end
        result[row, :] .= vec(scores)
    end
    return result
end

function decide(
    backend::NativeBackend,
    inputs::AbstractDict,
    questions::AbstractVector{<:AbstractQuestion},
)
    isempty(questions) && throw(ArgumentError("Provide at least one question."))
    counts = option_count.(questions)
    all(n -> n <= backend.max_options, counts) ||
        throw(ArgumentError("Question exceeds the checkpoint option limit."))
    scores = logits(backend, inputs)
    size(scores, 1) == length(questions) ||
        throw(ArgumentError("Questions must match the batch size."))
    return [
        answer(q, probabilities(view(scores, i, 1:n), backend.temperature)) for
        (i, (q, n)) in enumerate(zip(questions, counts))
    ]
end

decide(backend::NativeBackend, inputs::AbstractDict, question::AbstractQuestion) =
    only(decide(backend, inputs, [question]))
