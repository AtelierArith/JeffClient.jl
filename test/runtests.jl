using Test
using JeffClient
import JSON

const MODEL = joinpath(@__DIR__, "fixtures", "logits.onnx")

@testset "ONNX CPU inference and Jeff answers" begin
    backend = ONNXBackend(MODEL)
    try
        questions = [
            ChoiceQuestion(["a" => "A", "b" => "B", "c" => "C"]),
            NoulQuestion(),
            ScoreQuestion(["low", "medium", "high"]),
        ]
        # Non-square batch checks ONNX/Julia dimension and storage conventions.
        # Padded columns must not contribute to softmax, even when nonfinite.
        logits = Float32[0 log(2) log(3) 1000; 0 log(3) NaN Inf; 0 log(2) 0 -Inf]
        answers = decide(backend, Dict("scores" => logits), questions)
        @test answers[1].choice == "c"
        @test answers[1].probabilities["a"] ≈ 1 / 6 rtol=1e-6
        @test answers[1].probabilities["b"] ≈ 1 / 3 rtol=1e-6
        @test answers[1].probabilities["c"] ≈ 1 / 2 rtol=1e-6
        @test answers[1].confidence ≈ 0.25 rtol=1e-6
        @test answers[2].type == "noul"
        @test answers[2].noul ≈ 0.75 rtol=1e-6
        @test answers[3].score ≈ 1.0
        @test answers[3].confidence ≈ 0.25 rtol=1e-6
        @test answers[3].legend == Dict("0" => "low", "1" => "medium", "2" => "high")

        singleton = decide(
            backend,
            Dict("scores" => Float32[0 100]),
            ChoiceQuestion(["only" => "Only option"]),
        )
        @test singleton.choice == "only"
        @test singleton.confidence == 1.0

        tied = decide(
            backend,
            Dict("scores" => zeros(Float32, 1, 2)),
            ChoiceQuestion(["first" => "First", "second" => "Second"]),
        )
        @test tied.choice == "first"
        @test tied.confidence == 0.0

        @test_throws ArgumentError decide(backend, Dict("wrong" => logits), questions)
        @test_throws ArgumentError decide(
            backend,
            Dict("scores" => logits),
            [NoulQuestion()],
        )
        @test_throws ArgumentError decide(
            backend,
            Dict("scores" => zeros(Float32, 1, 1)),
            NoulQuestion(),
        )
        @test_throws ArgumentError decide(
            backend,
            Dict("scores" => Float32[NaN 0]),
            NoulQuestion(),
        )
        @test_throws ArgumentError decide(
            backend,
            Dict("scores" => Float32[Inf 0]),
            NoulQuestion(),
        )
        @test_throws ArgumentError decide(
            backend,
            Dict("scores" => logits),
            ChoiceQuestion[],
        )
    finally
        close(backend)
    end
end

@testset "Native Qwen versus independent PyTorch reference" begin
    path = joinpath(@__DIR__, "fixtures", "native")
    backend = NativeBackend(path)
    rows_to_matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    for sample in JSON.parsefile(joinpath(path, "reference.json"))
        inputs =
            Dict(name => rows_to_matrix(rows, Int64) for (name, rows) in sample["inputs"])
        expected = rows_to_matrix(sample["logits"], Float32)
        actual = logits(backend, inputs)
        @test actual ≈ expected atol = 2e-5 rtol = 2e-5
        questions = [ChoiceQuestion(["a" => "A", "b" => "B", "c" => "C"]), NoulQuestion()]
        answers = decide(backend, inputs, questions)
        @test answers[1].choice == ["a", "b", "c"][argmax(expected[1, :])]
        weights = exp.((expected[2, 1:2] .- maximum(expected[2, 1:2])) ./ 1.25)
        @test answers[2].noul ≈ weights[2] / sum(weights) atol = 2e-5
    end
    @test_throws ArgumentError logits(backend, Dict("input_ids" => ones(Int64, 1, 2)))
    @test_throws ArgumentError logits(
        backend,
        Dict("input_ids" => ones(Int64, 1, 2), "attention_mask" => zeros(Int64, 1, 2)),
    )
    @test_throws ArgumentError logits(
        backend,
        Dict("input_ids" => fill(Int64(64), 1, 2), "attention_mask" => ones(Int64, 1, 2)),
    )
end

@testset "Calibration and checkpoint limits" begin
    backend = ONNXBackend(MODEL; temperature = 2, max_options = 2)
    try
        q = ChoiceQuestion(["no" => "No", "yes" => "Yes"])
        result = decide(backend, Dict("scores" => reshape(Float32[0, log(9)], 1, 2)), q)
        @test result.probabilities["yes"] ≈ 0.75 rtol=1e-6
        @test result.confidence ≈ 0.5 rtol=1e-6
        @test_throws ArgumentError decide(
            backend,
            Dict("scores" => zeros(Float32, 1, 3)),
            ChoiceQuestion(["a" => "A", "b" => "B", "c" => "C"]),
        )
    finally
        close(backend)
    end
    backend = ONNXBackend(MODEL; temperature = 1e-300)
    try
        result = decide(backend, Dict("scores" => Float32[1e30 1e30]), NoulQuestion())
        @test result.noul == 0.5
    finally
        close(backend)
    end
end

@testset "Invalid configuration" begin
    @test_throws ArgumentError ChoiceQuestion(Pair{String,String}[])
    @test_throws ArgumentError ChoiceQuestion(["same" => "A", "same" => "B"])
    @test_throws ArgumentError ScoreQuestion(["one"])
    @test_throws ArgumentError ONNXBackend("missing.onnx")
    @test_throws ArgumentError ONNXBackend(MODEL; output_name = "missing")
    @test_throws ArgumentError ONNXBackend(MODEL; execution_provider = :coreml)
    for temperature in (0, -1, Inf, NaN)
        @test_throws ArgumentError ONNXBackend(MODEL; temperature)
    end
    @test_throws ArgumentError ONNXBackend(MODEL; max_options = 0)
    @test_throws ArgumentError ONNXBackend(MODEL; max_options = 256)
end

@testset "Export metadata calibration" begin
    mktempdir() do dir
        cp(MODEL, joinpath(dir, "model.onnx"))
        write(
            joinpath(dir, "export_config.json"),
            JSON.json(
                Dict(
                    "format_version" => 1,
                    "model_file" => "model.onnx",
                    "output_name" => "logits",
                ),
            ),
        )
        write(
            joinpath(dir, "decision_config.json"),
            JSON.json(
                Dict("format_version" => 1, "temperature" => 2.0, "max_options" => 2),
            ),
        )
        backend = load_export(dir)
        try
            result = decide(backend, Dict("scores" => Float32[0 log(9)]), NoulQuestion())
            @test result.noul ≈ 0.75 rtol=1e-6
            @test backend.max_options == 2
        finally
            close(backend)
        end
        write(
            joinpath(dir, "export_config.json"),
            JSON.json(
                Dict(
                    "format_version" => 1,
                    "model_file" => "../model.onnx",
                    "output_name" => "logits",
                ),
            ),
        )
        @test_throws ArgumentError load_export(dir)
    end
end
