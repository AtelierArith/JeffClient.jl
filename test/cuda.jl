# Optional hardware suite: run with CUDA and JeffClient in the active environment.
using Test, CUDA, JeffClient
const JSON = JeffClient.JSON
CUDA.functional(true) || error("The CUDA test suite requires a functional GPU.")
CUDA.allowscalar(false)

function matrix(rows, T)
    return permutedims(hcat([T.(row) for row in rows]...))
end
function inputs_for(sample)
    return Dict(name => matrix(rows, Int64) for (name, rows) in sample["inputs"])
end

@testset "Native CUDA independent reference and scratch ownership" begin
    path = joinpath(@__DIR__, "fixtures", "native")
    backend = NativeBackend(path; device = :cuda)
    @testset "cuBLAS coefficient reuse" begin
        a = CUDA.CuArray(Float32[1 2 3; 4 5 6])
        b = CUDA.CuArray(Float32[2 3 4; 5 6 7])
        for (left, right) in ((a, transpose(b)), (transpose(a), b))
            expected =
                left === a ? Array(a)*transpose(Array(b)) : transpose(Array(a))*Array(b)
            actual = JeffClient.native_forward_scope(backend.embedding) do
                Array(JeffClient.native_matmul(left, right))
            end
            @test actual ≈ expected
        end
    end
    cases = vcat(
        JSON.parsefile(joinpath(path, "reference.json")),
        JSON.parsefile(joinpath(path, "cuda_reference.json")),
    )
    retained = nothing
    retained_copy = nothing
    for sample in cases
        inputs = inputs_for(sample)
        saved = deepcopy(inputs)
        expected = matrix(sample["logits"], Float32)
        actual = JeffClient.logits(backend, inputs)
        @test actual ≈ expected atol=2e-5 rtol=2e-5
        @test inputs == saved
        GC.gc(true)
        @test JeffClient.logits(backend, inputs) ≈ expected atol=2e-5 rtol=2e-5
        retained !== nothing && @test retained == retained_copy
        retained, retained_copy = actual, copy(actual)
        masks = inputs["attention_mask"]
        for row in axes(masks, 1)
            expected_start =
                get(ENV, "JEFF_CUDA_TRIM_PADDING", "0") == "1" ?
                findfirst(==(1), view(masks, row, :)) : 1
            @test JeffClient.native_sequence_start(backend.embedding, masks, row) ==
                  expected_start
        end
    end
    inputs = inputs_for(last(cases))
    expected = matrix(last(cases)["logits"], Float32)
    tasks = [Threads.@spawn JeffClient.logits(backend, inputs) for _ = 1:2]
    for task in tasks
        @test fetch(task) ≈ expected atol=2e-5 rtol=2e-5
    end
    # A failed forward may have queued GPU work; subsequent reuse must be safe.
    @test_throws ErrorException JeffClient.native_forward_scope(backend.embedding) do
        buffer = JeffClient.native_matmul(backend.embedding, transpose(backend.embedding))
        error("injected failure after queued matmul")
    end
    @test JeffClient.logits(backend, inputs) ≈ expected atol=2e-5 rtol=2e-5
    if length(CUDA.devices()) >= 2
        original = CUDA.device()
        CUDA.device!(first(device for device in CUDA.devices() if device != original))
        @test JeffClient.logits(backend, inputs) ≈ expected atol=2e-5 rtol=2e-5
        @test CUDA.device() != CUDA.device(backend.embedding)
        CUDA.device!(original)
    end
end
