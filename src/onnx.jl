abstract type AbstractDecisionBackend end

"""
    ONNXBackend(path; execution_provider=:cpu, output_name="logits",
                temperature=1.0, max_options=255, provider_options=(;))
    ONNXBackend(f, path; kwargs...)

Load a model that returns uncalibrated option logits with shape
`(batch, options)`. Input tensors are passed through unchanged; this API does
not tokenize text or load Hugging Face checkpoints. Set `temperature` and
`max_options` to the values from the exported checkpoint's decision_config.json.

CPU and CUDA are supported by ONNXRunTime.jl. For CUDA, first import CUDA and
cuDNN in your environment. Model operator support depends on the provider.
Call `close(backend)` to release the inference session explicitly.
Alternatively, use a `do` block to close the session automatically, even if the
block throws an exception. The block's return value is returned:

```julia
ONNXBackend("jeff.onnx"; temperature=1.0) do backend
    decide(backend, inputs, question)
end
```
"""
struct ONNXBackend{S} <: AbstractDecisionBackend
    session::S
    output_name::String
    temperature::Float64
    max_options::Int
end

function ONNXBackend(
    path::AbstractString;
    execution_provider::Symbol = :cpu,
    output_name::AbstractString = "logits",
    temperature::Real = 1.0,
    max_options::Integer = 255,
    provider_options::NamedTuple = (;),
)
    isfile(path) || throw(ArgumentError("ONNX model does not exist: $path"))
    execution_provider in (:cpu, :cuda) || throw(ArgumentError("Use :cpu or :cuda."))
    scale = Float64(temperature)
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("Temperature must be positive and finite."))
    1 <= max_options <= 255 ||
        throw(ArgumentError("max_options must be between 1 and 255."))
    session = ONNXRunTime.load_inference(path; execution_provider, provider_options)
    if !(output_name in ONNXRunTime.output_names(session))
        names = ONNXRunTime.output_names(session)
        ONNXRunTime.release(session)
        throw(ArgumentError("Output '$output_name' is missing; available outputs: $names"))
    end
    return ONNXBackend(session, String(output_name), scale, Int(max_options))
end

Base.close(backend::ONNXBackend) = ONNXRunTime.release(backend.session)

function ONNXBackend(f::F, path::AbstractString; kwargs...) where {F}
    backend = ONNXBackend(path; kwargs...)
    try
        return f(backend)
    finally
        close(backend)
    end
end

"""
    decide(backend, inputs, questions)

Run prepared ONNX input tensors and return one Jeff-compatible answer per
question. Each question corresponds to a row of `(batch, options)` logits;
option order must match the export. Unused columns are excluded before softmax.
`inputs` is a dictionary with string keys and array values, in the ONNX model's
logical dimension order (for example `(batch, sequence)` for input IDs).

For a single question, `decide(backend, inputs, question)` returns one answer.
"""
function decide(
    backend::ONNXBackend,
    inputs::AbstractDict{<:AbstractString},
    questions::AbstractVector{<:AbstractQuestion},
)
    isempty(questions) && throw(ArgumentError("Provide at least one question."))
    counts = option_count.(questions)
    all(n -> n <= backend.max_options, counts) || throw(
        ArgumentError(
            "A question exceeds this checkpoint's $(backend.max_options)-option limit.",
        ),
    )
    expected = Set(ONNXRunTime.input_names(backend.session))
    Set(String.(keys(inputs))) == expected || throw(
        ArgumentError("Input names must match the model: $(sort!(collect(expected)))"),
    )
    all(value -> value isa AbstractArray, values(inputs)) ||
        throw(ArgumentError("Every ONNX input must be an array."))
    outputs =
        backend.session(Dict(String(k) => v for (k, v) in inputs), [backend.output_name])
    logits = outputs[backend.output_name]
    logits isa AbstractMatrix{<:Real} ||
        throw(ArgumentError("Model output must be a real (batch, options) matrix."))
    size(logits, 1) == length(questions) ||
        throw(ArgumentError("Logit rows must match the number of questions."))
    maximum(counts) <= size(logits, 2) ||
        throw(ArgumentError("Model output has fewer columns than a question's options."))
    return [
        answer(q, probabilities(view(logits, i, 1:n), backend.temperature)) for
        (i, (q, n)) in enumerate(zip(questions, counts))
    ]
end

decide(
    backend::ONNXBackend,
    inputs::AbstractDict{<:AbstractString},
    question::AbstractQuestion,
) = only(decide(backend, inputs, [question]))
