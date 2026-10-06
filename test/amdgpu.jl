# Optional hardware suite: run with AMDGPU and JeffClient in the active
# environment, whose QwenDecisionCore checkout must provide the AMDGPU
# extension. On a machine without an AMD GPU it is skipped by the functional
# guard instead of failing.
using Test, AMDGPU, JeffClient
import QwenDecisionCore
const JSON = JeffClient.JSON
AMDGPU.functional() || error("The AMDGPU test suite requires a functional GPU.")
AMDGPU.allowscalar(false)

function matrix(rows, T)
    return permutedims(hcat([T.(row) for row in rows]...))
end
function inputs_for(sample)
    return Dict(name => matrix(rows, Int64) for (name, rows) in sample["inputs"])
end

@testset "Native AMDGPU independent reference and scratch ownership" begin
    path = joinpath(@__DIR__, "fixtures", "native")
    backend = NativeBackend(path; device = :amdgpu)
    @testset "rocBLAS coefficient reuse" begin
        a = AMDGPU.ROCArray(Float32[1 2 3; 4 5 6])
        b = AMDGPU.ROCArray(Float32[2 3 4; 5 6 7])
        for (left, right) in ((a, transpose(b)), (transpose(a), b))
            expected =
                left === a ? Array(a)*transpose(Array(b)) : transpose(Array(a))*Array(b)
            actual = QwenDecisionCore.native_forward_scope(backend.backbone.embedding) do
                Array(QwenDecisionCore.native_matmul(left, right))
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
        trim = get(ENV, "QDC_AMDGPU_TRIM_PADDING", "1") == "1"
        for row in axes(masks, 1)
            expected_start = trim ? findfirst(==(1), view(masks, row, :)) : 1
            @test QwenDecisionCore.native_sequence_start(
                backend.backbone.embedding,
                masks,
                row,
            ) == expected_start
        end
    end
    inputs = inputs_for(last(cases))
    expected = matrix(last(cases)["logits"], Float32)
    tasks = [Threads.@spawn JeffClient.logits(backend, inputs) for _ = 1:2]
    for task in tasks
        @test fetch(task) ≈ expected atol=2e-5 rtol=2e-5
    end
    # A failed forward may have queued GPU work; subsequent reuse must be safe.
    @test_throws ErrorException QwenDecisionCore.native_forward_scope(
        backend.backbone.embedding,
    ) do
        buffer = QwenDecisionCore.native_matmul(
            backend.backbone.embedding,
            transpose(backend.backbone.embedding),
        )
        error("injected failure after queued matmul")
    end
    @test JeffClient.logits(backend, inputs) ≈ expected atol=2e-5 rtol=2e-5
end
