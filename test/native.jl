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
        JeffClient.with_cpu_settings(
            :delta_workspace => "1",
            :parallel_heads => parallel,
            :mlp_workspace => mlp,
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
    JeffClient.with_cpu_settings(:final_query => "1") do
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
        JeffClient.with_cpu_settings(
            :delta_chunk_size => string(chunk),
            :delta_workers => string(workers),
            :delta_workspace => "1",
            :parallel_heads => "1",
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
    JeffClient.with_cpu_settings(:delta_chunk_size => "0") do
        @test_throws ArgumentError JeffClient.cpu_delta_worker_workspace(backend.config, 9)
    end
    JeffClient.with_cpu_settings(:delta_workers => "0") do
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
        expected = JeffClient.with_cpu_settings(:recurrent_delta => "0") do
            JeffClient.delta_attention(attention, x, mask, cfg)
        end
        for parallel in ("0", "1")
            actual = JeffClient.with_cpu_settings(
                :recurrent_delta => "1",
                :parallel_heads => parallel,
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
        JeffClient.with_cpu_settings(:parallel_projections => "1") do
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

@testset "CPU projection thread policy task-local lifetime" begin
    key = :jeff_cpu_projection_blas_threads
    JeffClient.with_cpu_settings(
        :parallel_projections => "1",
        :projection_thread_scope => "1",
    ) do
        saved = get(task_local_storage(), key, nothing)
        actual = JeffClient.BLAS.get_num_threads()
        @test JeffClient.cpu_projection_scope(
            () -> JeffClient.cpu_projection_blas_threads(),
        ) == actual
        @test get(task_local_storage(), key, nothing) === saved
        @test_throws ErrorException JeffClient.cpu_projection_scope(
            () -> error("policy lifetime test"),
        )
        @test get(task_local_storage(), key, nothing) === saved
        task_local_storage(key, 123) do
            @test JeffClient.cpu_projection_scope(
                () -> JeffClient.cpu_projection_blas_threads(),
            ) == actual
            @test task_local_storage(key) == 123
            tasks = [
                Threads.@spawn JeffClient.cpu_projection_scope(
                    () -> JeffClient.cpu_projection_blas_threads(),
                ) for _ = 1:2
            ]
            @test all(fetch(task) == actual for task in tasks)
            @test task_local_storage(key) == 123
        end
        @test get(task_local_storage(), key, nothing) === saved
    end
end

@testset "CPU full-attention head ownership and mask" begin
    backend = NativeBackend(joinpath(@__DIR__, "fixtures", "native"))
    attention = last(backend.layers).attention
    cfg = backend.config
    previous_threads = JeffClient.BLAS.get_num_threads()
    try
        JeffClient.BLAS.set_num_threads(1)
        for n in (1, 9, 65, 129), holes in (false, true)
            x = reshape(sin.(Float32.(1:(cfg.hidden*n))), cfg.hidden, n)
            mask = ones(Int64, n)
            holes && n > 1 && (mask[1:3:(n-1)] .= 0)
            saved_x, saved_mask = copy(x), copy(mask)
            expected = JeffClient.with_cpu_settings(:parallel_full_heads => "0") do
                JeffClient.full_attention(attention, x, mask, cfg)
            end
            JeffClient.with_cpu_settings(:parallel_full_heads => "1") do
                tasks = [
                    Threads.@spawn JeffClient.full_attention(attention, x, mask, cfg)
                    for _ = 1:2
                ]
                @test all(
                    isapprox(fetch(task), expected; atol = 2e-5, rtol = 2e-5) for
                    task in tasks
                )
                GC.gc(true)
                @test JeffClient.full_attention(attention, x, mask, cfg) ≈ expected atol=2e-5 rtol=2e-5
                @test x == saved_x
                @test mask == saved_mask
            end
        end
    finally
        JeffClient.BLAS.set_num_threads(previous_threads)
    end
end

@testset "CPU Delta normalization loop" begin
    for width in (1, 7, 16, 128),
        heads in (1, 3),
        n in (0, 1, 65),
        scale in (1.0f0, sqrt(Float32(width)))

        original = reshape(sin.(Float32.(1:(width*heads*n))), width, heads, n)
        expected = original ./ (sqrt.(sum(abs2, original; dims = 1) .+ 1.0f-6) .* scale)
        JeffClient.with_cpu_settings(:delta_norm_loop => "1") do
            actual = copy(original)
            @test JeffClient.cpu_normalize_delta_heads!(actual, scale) === actual
            @test actual ≈ expected atol=2e-6 rtol=2e-6
        end
    end
    for x in
        (0.0f0, -0.0f0, Float32(NaN), Float32(Inf), floatmax(Float32), nextfloat(0.0f0))
        original = fill(x, 8, 2, 3)
        expected = original ./ sqrt.(sum(abs2, original; dims = 1) .+ 1.0f-6)
        actual = JeffClient.with_cpu_settings(:delta_norm_loop => "1") do
            JeffClient.cpu_normalize_delta_heads!(copy(original), 1.0f0)
        end
        @test isequal(actual, expected)
    end
end

@testset "CPU Delta projection buffers overwrite and mask reuse" begin
    backend = NativeBackend(joinpath(@__DIR__, "fixtures", "native"))
    cfg = backend.config
    attention = first(l.attention for l in backend.layers if l.attention.kind == :delta)
    for n in (1, 9, 65, 129), recurrent in ("0", "1"), parallel in ("0", "1")
        x = reshape(sin.(Float32.(1:(cfg.hidden*n))), cfg.hidden, n)
        saved = copy(x)
        JeffClient.with_cpu_settings(
            :delta_projection_workspace=>"1",
            :recurrent_delta=>recurrent,
            :parallel_heads=>parallel,
        ) do
            buffers = JeffClient.cpu_delta_projection_workspace(cfg, n)
            state = JeffClient.cpu_delta_workspace(cfg, n)
            for holes in (false, true, false)
                mask = ones(Int64, n)
                holes && n > 1 && (mask[1:3:(n-1)] .= 0)
                expected =
                    JeffClient.delta_attention(attention, x, mask, cfg, state, nothing)
                for array in values(buffers)
                    fill!(array, Float32(NaN))
                end
                GC.gc(true)
                actual = JeffClient.delta_attention(attention, x, mask, cfg, state, buffers)
                @test actual === buffers.projected
                @test actual ≈ expected atol=2e-5 rtol=2e-5
                @test x == saved
            end
            @test_throws DimensionMismatch JeffClient.delta_attention(
                attention,
                x[:, 1:0],
                Int64[],
                cfg,
                state,
                buffers,
            )
        end
    end
end
@testset "CPU projection accumulation and MLP residual ownership" begin
    old_threads = JeffClient.BLAS.get_num_threads()
    try
        JeffClient.BLAS.set_num_threads(1)
        for parallel in ("0", "1")
            JeffClient.with_cpu_settings(:parallel_projections => parallel) do
                for n in (1, 9, 65)
                    weight = reshape(sin.(Float32.(1:(512*512))), 512, 512) ./ 512
                    x = reshape(cos.(Float32.(1:(512*n))), 512, n)
                    residual = sin.(x)
                    saved_x, saved_w = copy(x), copy(weight)
                    expected = residual .+ transpose(weight) * x
                    @test JeffClient.cpu_projection!(residual, weight, x, 1.0f0) ===
                          residual
                    @test residual ≈ expected atol=2e-5 rtol=2e-5
                    @test x == saved_x
                    @test weight == saved_w
                end
                weight = reshape(sin.(Float32.(1:64)), 8, 8)
                x = copy(weight)
                expected = x .+ transpose(weight) * x
                @test JeffClient.cpu_projection!(x, weight, x, 1.0f0) ≈ expected
                x = copy(weight)
                expected = weight .+ transpose(weight) * x
                @test JeffClient.cpu_projection!(weight, weight, x, 1.0f0) ≈ expected
            end
        end
        backend = NativeBackend(joinpath(@__DIR__, "fixtures", "native"))
        mlp = first(backend.layers).mlp
        for n in (1, 9, 65), workspace in (false, true)
            x = reshape(
                sin.(Float32.(1:(backend.config.hidden*n))),
                backend.config.hidden,
                n,
            )
            residual = cos.(x)
            saved = copy(x)
            expected = residual .+ JeffClient.native_mlp(mlp, x)
            width = size(mlp.gate, 2)
            buffers = workspace ? (fill(NaN32, width, n), fill(NaN32, width, n)) : nothing
            JeffClient.with_cpu_settings(:mlp_residual_fusion => "1") do
                @test JeffClient.cpu_mlp_add!(residual, mlp, x, buffers) === residual
                @test residual ≈ expected atol=2e-5 rtol=2e-5
                @test x == saved
            end
        end
    finally
        JeffClient.BLAS.set_num_threads(old_threads)
    end
end
@testset "Automatic CPU policy and scoped diagnostics" begin
    settings = JeffClient.cpu_settings()
    withenv("JEFF_CPU_DELTA_CHUNK_SIZE" => "invalid", "JEFF_CPU_PARALLEL_HEADS" => "0") do
        @test JeffClient.cpu_settings() == settings
        @test JeffClient.cpu_delta_chunk_size() == settings.delta_chunk_size
        @test JeffClient.cpu_setting(:parallel_heads)
    end
    JeffClient.with_cpu_settings(:parallel_heads => false) do
        @test !JeffClient.cpu_setting(:parallel_heads)
    end
    @test JeffClient.cpu_settings() == settings
    @test_throws ErrorException JeffClient.with_cpu_settings(:parallel_heads => false) do
        error("scope restoration")
    end
    @test JeffClient.cpu_settings() == settings
    @test settings.mlp_workspace && settings.trim_padding
end
