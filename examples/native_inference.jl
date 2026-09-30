using JeffClient
import JSON

if length(ARGS) == 2 && ARGS[2] == "metal"
    import Metal
end

function main()
    length(ARGS) <= 2 || error("Usage: native_inference.jl [CHECKPOINT] [cpu|metal]")
    sample = JSON.parsefile(joinpath(@__DIR__, "data", "parcel.json"))
    device = length(ARGS) == 2 ? Symbol(ARGS[2]) : :cpu
    device in (:cpu, :metal) || error("Choose cpu or metal.")
    source = isempty(ARGS) ? sample["model"] : ARGS[1]
    checkpoint = resolve_checkpoint(source; revision = sample["revision"])
    backend = NativeBackend(checkpoint; device)
    raw = sample["question"]
    question = ChoiceQuestion(
        # Keep the option order used when preparing these prompt tokens.
        [key => String(raw["criteria"][key]) for key in ("refund", "delivery")];
        instructions = raw["instructions"],
    )
    # These real prompt tokens were prepared with the pinned checkpoint's
    # original Jeff tokenizer. Changing the text alone does not change tokens.
    inputs = Dict(
        name => reduce(vcat, [permutedims(Int64.(row)) for row in rows]) for
        (name, rows) in sample["inputs"]
    )
    result = decide(backend, inputs, question)
    println("Device: ", device)
    println("Input: ", sample["state"])
    println("Question: ", question.instructions)
    println("Choice: ", result.choice)
    for (key, _) in question.criteria
        println("  ", key, ": ", round(result.probabilities[key]; digits = 6))
    end
    println("Confidence: ", round(result.confidence; digits = 6))
end

main()
