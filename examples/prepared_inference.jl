using JeffClient

# A runnable example of the inference boundary. This fixture is an Identity
# graph, not a Jeff checkpoint: scores are supplied here to show postprocessing.
model = joinpath(@__DIR__, "..", "test", "fixtures", "logits.onnx")
question = ChoiceQuestion(["no" => "No", "yes" => "Yes"])
inputs = Dict("scores" => reshape(Float32[0, log(9)], 1, 2))
ONNXBackend(model; temperature = 2.0, max_options = 2) do backend
    result = decide(backend, inputs, question)
    println("Choice: ", result.choice)
    println("Probabilities: ", result.probabilities)
    println("Confidence: ", result.confidence)
end
