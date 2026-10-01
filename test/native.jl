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

@testset "CPU forward-local scratch ownership" begin
    backend = NativeBackend(joinpath(@__DIR__, "fixtures", "native"))
    samples = JSON.parsefile(joinpath(@__DIR__, "fixtures", "native", "reference.json"))
    matrix(rows) = reduce(vcat, [permutedims(Int64.(row)) for row in rows])
    for parallel in ("0", "1"), mlp in ("0", "1")
        withenv(
            "JEFF_CPU_DELTA_WORKSPACE" => "1",
            "JEFF_CPU_PARALLEL_HEADS" => parallel,
            "JEFF_CPU_MLP_WORKSPACE" => mlp,
        ) do
            for sample in samples
                inputs = Dict(name => matrix(rows) for (name, rows) in sample["inputs"])
                saved = deepcopy(inputs)
                expected =
                    reduce(vcat, [permutedims(Float32.(row)) for row in sample["logits"]])
                first_result = logits(backend, inputs)
                retained = copy(first_result)
                GC.gc(true)
                tasks = [Threads.@spawn(logits(backend, inputs)) for _ = 1:2]
                for task in tasks
                    @test fetch(task) ≈ expected atol=2e-5 rtol=2e-5
                end
                @test first_result ≈ expected atol=2e-5 rtol=2e-5
                @test first_result == retained
                @test inputs == saved
            end
        end
    end
end

@testset "CPU final query preserves full-context attention" begin
    backend = NativeBackend(joinpath(@__DIR__, "fixtures", "native"))
    layer = last(backend.layers)
    cfg = backend.config
    for n in (1, 9, 65), holes in (false, true)
        x = reshape(sin.(Float32.(1:(cfg.hidden*n))), cfg.hidden, n)
        mask = ones(Int64, n)
        holes && n > 1 && (mask[1:2:(n-1)] .= 0)
        expected = JeffClient.full_attention(layer.attention, x, mask, cfg)[:, end:end]
        actual = JeffClient.cpu_final_full_attention(layer.attention, x, mask, cfg)
        @test actual ≈ expected atol=2e-5 rtol=2e-5
    end
    withenv("JEFF_CPU_FINAL_QUERY" => "1") do
        for sample in
            JSON.parsefile(joinpath(@__DIR__, "fixtures", "native", "reference.json"))
            inputs = Dict(
                name => reduce(vcat, [permutedims(Int64.(row)) for row in rows]) for
                (name, rows) in sample["inputs"]
            )
            expected =
                reduce(vcat, [permutedims(Float32.(row)) for row in sample["logits"]])
            @test logits(backend, inputs) ≈ expected atol=2e-5 rtol=2e-5
        end
    end
end

@testset "CPU Delta chunk and worker tuning" begin
    backend = NativeBackend(joinpath(@__DIR__, "fixtures", "native"))
    samples = JSON.parsefile(joinpath(@__DIR__, "fixtures", "native", "reference.json"))
    for chunk in (1, 3, 16, 32, 64, 128), workers in (1, 2, 4, 8)
        withenv(
            "JEFF_CPU_DELTA_CHUNK_SIZE" => string(chunk),
            "JEFF_CPU_DELTA_WORKERS" => string(workers),
            "JEFF_CPU_DELTA_WORKSPACE" => "1",
            "JEFF_CPU_PARALLEL_HEADS" => "1",
        ) do
            for sample in samples
                inputs = Dict(
                    name => reduce(vcat, [permutedims(Int64.(row)) for row in rows]) for
                    (name, rows) in sample["inputs"]
                )
                expected =
                    reduce(vcat, [permutedims(Float32.(row)) for row in sample["logits"]])
                @test logits(backend, inputs) ≈ expected atol=2e-5 rtol=2e-5
            end
        end
    end
    withenv("JEFF_CPU_DELTA_CHUNK_SIZE" => "0") do
        @test_throws ArgumentError JeffClient.cpu_delta_worker_workspace(backend.config, 9)
    end
    withenv("JEFF_CPU_DELTA_WORKERS" => "0") do
        @test_throws ArgumentError JeffClient.cpu_delta_workers(backend.config)
    end
end

@testset "CPU recurrent Delta versus chunked reference" begin
    backend = NativeBackend(joinpath(@__DIR__, "fixtures", "native"))
    attention =
        first(layer.attention for layer in backend.layers if layer.attention.kind == :delta)
    cfg = backend.config
    for n in (1, 9, 65, 129, 256), holes in (false, true)
        x = reshape(sin.(Float32.(1:(cfg.hidden*n))), cfg.hidden, n)
        mask = ones(Int64, n)
        holes && n > 1 && (mask[1:3:(n-1)] .= 0)
        expected = withenv("JEFF_CPU_RECURRENT_DELTA" => "0") do
            JeffClient.delta_attention(attention, x, mask, cfg)
        end
        for parallel in ("0", "1")
            actual = withenv(
                "JEFF_CPU_RECURRENT_DELTA" => "1",
                "JEFF_CPU_PARALLEL_HEADS" => parallel,
            ) do
                scratch = JeffClient.cpu_delta_workspace(cfg, n)
                JeffClient.delta_attention(attention, x, mask, cfg, scratch)
            end
            @test actual ≈ expected atol=2e-5 rtol=2e-5
        end
    end
end

@testset "CPU parallel projection ownership" begin
    previous_threads = JeffClient.BLAS.get_num_threads()
    try
        JeffClient.BLAS.set_num_threads(1)
        withenv("JEFF_CPU_PARALLEL_PROJECTIONS" => "1") do
            weight = reshape(sin.(Float32.(1:(64*257))), 64, 257)
            for w in (weight, transpose(permutedims(weight))), n in (1, 65)
                input = reshape(cos.(Float32.(1:(64*n))), 64, n)
                saved_input, saved_weight = copy(input), copy(w)
                expected = transpose(w) * input
                output = fill(Float32(NaN), 257, n)
                @test JeffClient.cpu_projection!(output, w, input) === output
                @test output ≈ expected atol=2e-5 rtol=2e-5
                @test input == saved_input
                @test w == saved_weight
                tasks = [Threads.@spawn(JeffClient.native_linear(w, input)) for _ = 1:2]
                @test all(fetch(task) ≈ expected for task in tasks)
            end
            w = reshape(sin.(Float32.(1:(256*256))), 256, 256)
            input = copy(w)
            expected = transpose(w) * input
            @test JeffClient.cpu_projection!(input, w, input) ≈ expected
            input = copy(w)
            @test JeffClient.cpu_projection!(w, w, input) ≈ expected
            @test_throws DimensionMismatch JeffClient.cpu_projection!(
                zeros(Float32, 3, 2),
                zeros(Float32, 4, 3),
                zeros(Float32, 5, 2),
            )
        end
    finally
        JeffClient.BLAS.set_num_threads(previous_threads)
    end
end
