@testset "Native Qwen versus independent PyTorch reference" begin
    path = joinpath(@__DIR__, "fixtures", "native")
    backend = NativeBackend(path)
    rows_to_matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    for sample in JSON.parsefile(joinpath(path, "reference.json"))
        inputs =
            Dict(name => rows_to_matrix(rows, Int64) for (name, rows) in sample["inputs"])
        expected = rows_to_matrix(sample["logits"], Float32)
        actual = logits(backend, inputs)
        @test actual ≈ expected atol=2e-5 rtol=2e-5
        questions = [ChoiceQuestion(["a" => "A", "b" => "B", "c" => "C"]), NoulQuestion()]
        answers = decide(backend, inputs, questions)
        @test answers[1].choice == ["a", "b", "c"][argmax(expected[1, :])]
        weights = exp.((expected[2, 1:2] .- maximum(expected[2, 1:2])) ./ 1.25)
        @test answers[2].noul ≈ weights[2] / sum(weights) atol=2e-5
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
