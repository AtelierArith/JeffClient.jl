using JeffClient
import JSON

directory =
    isempty(ARGS) ? joinpath(@__DIR__, "..", "artifacts", "jeff-0.8b-onnx") : only(ARGS)
backend = load_export(directory)
samples = JSON.parsefile(joinpath(directory, "reference.json"))
matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
try
    for sample in samples
        inputs = Dict(name => matrix(rows, Int64) for (name, rows) in sample["inputs"])
        expected = matrix(sample["logits"], Float32)
        actual = backend.session(inputs)["logits"]
        max_error = maximum(abs.(actual .- expected))
        all(abs.(actual .- expected) .<= 2e-3 .+ 2e-3 .* abs.(expected)) ||
            error("Julia ONNX logits differ from the PyTorch reference.")
        raw = sample["question"]
        question =
            raw["type"] == "choice" ?
            ChoiceQuestion(
                [String(k) => String(v) for (k, v) in raw["criteria"]];
                instructions = raw["instructions"],
            ) : NoulQuestion(; instructions = raw["instructions"])
        result = decide(backend, inputs, fill(question, size(expected, 1)))
        println("Max logit error: ", max_error, "; answers: ", result)
    end
finally
    close(backend)
end
