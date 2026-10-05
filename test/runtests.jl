using Test
using JeffClient
import JSON

const FIXTURE = joinpath(@__DIR__, "fixtures", "native")
rows_to_matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])

@testset "Native Qwen versus independent PyTorch reference" begin
    backend = NativeBackend(FIXTURE)
    for sample in JSON.parsefile(joinpath(FIXTURE, "reference.json"))
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
    fixture = NativeBackend(FIXTURE)
    sample = first(JSON.parsefile(joinpath(FIXTURE, "reference.json")))
    inputs = Dict(name => rows_to_matrix(rows, Int64) for (name, rows) in sample["inputs"])
    scores = logits(fixture, inputs)
    # Same backbone and readout with a different temperature and option limit.
    backend = NativeBackend(fixture.backbone, fixture.readout, 2.0, 2)
    questions = [ChoiceQuestion(["no" => "No", "yes" => "Yes"]), NoulQuestion()]
    answers = decide(backend, inputs, questions)
    weights = exp.((scores[1, 1:2] .- maximum(scores[1, 1:2])) ./ 2)
    @test answers[1].probabilities["yes"] ≈ weights[2] / sum(weights) rtol = 1e-5
    # Options beyond the trained limit and question/batch mismatches are rejected.
    three = ChoiceQuestion(["a" => "A", "b" => "B", "c" => "C"])
    @test_throws ArgumentError decide(backend, inputs, [three, NoulQuestion()])
    @test_throws ArgumentError decide(backend, inputs, [NoulQuestion()])
    @test_throws ArgumentError decide(backend, inputs, ChoiceQuestion[])
end

@testset "Invalid configuration" begin
    @test_throws ArgumentError ChoiceQuestion(Pair{String,String}[])
    @test_throws ArgumentError ChoiceQuestion(["same" => "A", "same" => "B"])
    @test_throws ArgumentError ScoreQuestion(["one"])
    mktempdir() do dir
        for name in readdir(FIXTURE)
            cp(joinpath(FIXTURE, name), joinpath(dir, name))
        end
        for temperature in (0, -1)
            write(
                joinpath(dir, "decision_config.json"),
                JSON.json(
                    Dict(
                        "format_version" => 1,
                        "temperature" => temperature,
                        "max_options" => 3,
                    ),
                ),
            )
            @test_throws ArgumentError NativeBackend(dir)
        end
        for max_options in (0, 256)
            write(
                joinpath(dir, "decision_config.json"),
                JSON.json(
                    Dict(
                        "format_version" => 1,
                        "temperature" => 1.0,
                        "max_options" => max_options,
                    ),
                ),
            )
            @test_throws ArgumentError NativeBackend(dir)
        end
        write(
            joinpath(dir, "decision_config.json"),
            JSON.json(
                Dict("format_version" => 2, "temperature" => 1.0, "max_options" => 3),
            ),
        )
        @test_throws ArgumentError NativeBackend(dir)
    end
end
