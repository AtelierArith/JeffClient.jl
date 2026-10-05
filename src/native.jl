# Jeff's native decision backend: a QwenDecisionCore.QwenBackbone plus a linear
# readout. The backbone forward pass, the safetensors reader, the Hub resolver
# and the CPU policy live in QwenDecisionCore; this file keeps only what is
# Jeff-specific (decision_config.json, readout.safetensors and the answer
# batching).

# A Jeff checkpoint is a Qwen backbone plus its decision files.
const CHECKPOINT_REQUIRED =
    (QwenDecisionCore.DEFAULT_REQUIRED..., "decision_config.json", "readout.safetensors")

"""
    resolve_checkpoint(model_id_or_path; revision="main", kwargs...)

Return a complete local Jeff checkpoint directory, downloading missing files
from the Hugging Face Hub when needed. This is
`QwenDecisionCore.resolve_checkpoint` with Jeff's `decision_config.json` and
`readout.safetensors` added to the required files; see it for the cache
locations, offline mode and the other keywords.
"""
resolve_checkpoint(model_id_or_path::AbstractString; kwargs...) =
    QwenDecisionCore.resolve_checkpoint(
        model_id_or_path;
        required = CHECKPOINT_REQUIRED,
        kwargs...,
    )

"""
    NativeBackend(checkpoint; device=:cpu)

Load Jeff's text-only Qwen3.5 backbone and readout directly from a local
checkpoint directory or a Hugging Face repository ID. The complete forward
pass is implemented in Julia through `QwenDecisionCore`. Inputs are prepared
token IDs and masks; tokenization is not part of this API. Only float32
inference, default partial RoPE and bias-free projections are supported.

Use `device=:metal` after importing Metal, or `device=:cuda` after importing
CUDA, for GPU execution. CPU inference automatically selects its platform
policy through QwenDecisionCore.
"""
struct NativeBackend{B,R} <: AbstractDecisionBackend
    backbone::B
    readout::R
    temperature::Float64
    max_options::Int
end

function NativeBackend(checkpoint::AbstractString; device::Symbol = :cpu)
    directory = resolve_checkpoint(checkpoint)
    decision = JSON.parsefile(joinpath(directory, "decision_config.json"))
    decision["format_version"] == 1 ||
        throw(ArgumentError("Unsupported decision checkpoint format."))
    backbone = QwenBackbone(directory; device)
    # Materialize once; the reader returns a reinterpreted view of the file.
    readout = Matrix{Float32}(
        read_native_weights(joinpath(directory, "readout.safetensors"))["weight"],
    )
    temperature = Float64(decision["temperature"])
    isfinite(temperature) && temperature > 0 ||
        throw(ArgumentError("Invalid checkpoint temperature."))
    limit = Int(decision["max_options"])
    1 <= limit <= size(readout, 2) <= 255 ||
        throw(ArgumentError("Invalid trained option limit or readout."))
    return NativeBackend(backbone, readout, temperature, limit)
end

"""
    logits(backend::NativeBackend, inputs)

Compute uncalibrated `(batch, options)` scores from `input_ids` and
`attention_mask`, each with logical `(batch, sequence)` dimensions. Batches
are processed one sequence at a time unless an accelerator extension provides
a batched `QwenDecisionCore.batch_backbone_hidden` path. No generation or KV
cache is used.
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
    all(id -> 0 <= id < size(backend.backbone.embedding, 2), ids) ||
        throw(ArgumentError("Token ID is outside the vocabulary."))
    all(value -> value in (0, 1), mask) ||
        throw(ArgumentError("Mask values must be zero or one."))
    all(mask[:, end] .== 1) ||
        throw(ArgumentError("The final position must be active; use left padding."))
    readout = backend.readout
    batched = QwenDecisionCore.batch_backbone_hidden(backend.backbone, ids, mask)
    batched === nothing || return permutedims(transpose(readout) * batched)
    result = Matrix{Float32}(undef, size(ids, 1), size(readout, 2))
    for row in axes(ids, 1)
        first_token =
            QwenDecisionCore.native_sequence_start(backend.backbone.embedding, mask, row)
        state = backbone_last_hidden(
            backend.backbone,
            vec(ids[row, first_token:end]),
            vec(mask[row, first_token:end]),
        )
        result[row, :] .= transpose(readout) * state
    end
    return result
end

"""
    decide(backend::NativeBackend, inputs, questions)

Return one Jeff-compatible answer per question. Each question consumes one row
of the `(batch, options)` logits; option order must match the checkpoint.
"""
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
