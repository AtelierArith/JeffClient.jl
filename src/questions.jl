abstract type AbstractQuestion end

"""
    ChoiceQuestion(criteria; instructions="")

Construct a choice question from an ordered vector of string key => description
pairs. The order must match the option columns in the model's logits.
"""
struct ChoiceQuestion <: AbstractQuestion
    criteria::Vector{Pair{String,String}}
    instructions::String
    function ChoiceQuestion(
        criteria::AbstractVector{<:Pair};
        instructions::AbstractString = "",
    )
        1 <= length(criteria) <= 255 || throw(ArgumentError("Provide 1 to 255 options."))
        converted = Pair{String,String}[String(k) => String(v) for (k, v) in criteria]
        allunique(first.(converted)) || throw(ArgumentError("Option keys must be unique."))
        new(converted, String(instructions))
    end
end

"""
    NoulQuestion(; instructions="")

Construct a yes/no question. Model columns must be ordered false, true.
"""
struct NoulQuestion <: AbstractQuestion
    instructions::String
end
NoulQuestion(; instructions::AbstractString = "") = NoulQuestion(String(instructions))

"""
    ScoreQuestion(criteria; instructions="")

Construct a question with 2 to 10 ordered scale descriptions. Returned scores
use Jeff's zero-based scale, from 0 to length(criteria) - 1.
"""
struct ScoreQuestion <: AbstractQuestion
    criteria::Vector{String}
    instructions::String
    function ScoreQuestion(
        criteria::AbstractVector{<:AbstractString};
        instructions::AbstractString = "",
    )
        2 <= length(criteria) <= 10 || throw(ArgumentError("Provide 2 to 10 scale levels."))
        new(String.(criteria), String(instructions))
    end
end

option_count(q::ChoiceQuestion) = length(q.criteria)
option_count(::NoulQuestion) = 2
option_count(q::ScoreQuestion) = length(q.criteria)

function probabilities(logits::AbstractVector{<:Real}, temperature::Real)
    all(isfinite, logits) || throw(ArgumentError("Active option logits must be finite."))
    # Subtract before dividing to avoid overflow at small temperatures.
    weights = exp.((Float64.(logits) .- maximum(logits)) ./ temperature)
    weights ./ sum(weights)
end

function answer(q::ChoiceQuestion, values)
    best = argmax(values)
    n = length(values)
    confidence = n == 1 ? 1.0 : clamp((values[best] - 1 / n) / (1 - 1 / n), 0.0, 1.0)
    return (
        type = "choice",
        probabilities = Dict(first.(q.criteria) .=> values),
        choice = first(q.criteria[best]),
        confidence = confidence,
    )
end

answer(::NoulQuestion, values) = (type = "noul", noul = values[2])

function answer(q::ScoreQuestion, values)
    n = length(values)
    levels = 0:(n-1)
    best = argmax(values) - 1
    distance = sum(values .* abs.(levels .- best))
    midpoint = (n - 1) / 2
    baseline = sum(abs.(levels .- midpoint)) / n
    keys = string.(levels)
    return (
        type = "score",
        probabilities = Dict(keys .=> values),
        legend = Dict(keys .=> q.criteria),
        score = sum(levels .* values),
        confidence = max(0.0, 1.0 - distance / baseline),
    )
end
