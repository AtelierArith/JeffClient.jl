using JeffClient
import JSON
import Metal

function main()
    length(ARGS) <= 1 || error("Usage: metal_inference.jl [CHECKPOINT]")
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)

    sample = JSON.parsefile(joinpath(@__DIR__, "data", "parcel.json"))
    source = isempty(ARGS) ? sample["model"] : only(ARGS)
    checkpoint = resolve_checkpoint(source; revision = sample["revision"])
    backend = NativeBackend(checkpoint; device = :metal)

    raw = sample["question"]
    question = ChoiceQuestion(
        [key => String(raw["criteria"][key]) for key in ("refund", "delivery")];
        instructions = raw["instructions"],
    )
    # The bundled tokens encode this fixed prompt and the refund/delivery order.
    # Changing the displayed text alone does not regenerate model inputs.
    inputs = Dict(
        name => reduce(vcat, [permutedims(Int64.(row)) for row in rows]) for
        (name, rows) in sample["inputs"]
    )

    result = decide(backend, inputs, question)
    Metal.synchronize()
    println("Device: metal")
    println("Input: ", sample["state"])
    println("Question: ", question.instructions)
    println("Choice: ", result.choice)
    for (key, _) in question.criteria
        println("  ", key, ": ", round(result.probabilities[key]; digits = 6))
    end
    println("Confidence: ", round(result.confidence; digits = 6))
end

main()
