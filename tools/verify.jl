# Compare NativeBackend logits on CPU, Metal, CUDA or AMDGPU with an
# independent PyTorch reference (tools/build_reference.jl). Every case runs
# twice with a full GC in between, so reused device buffers and retained
# results are checked as well as a fresh forward.
#
#   julia --threads=8 --project=tools tools/verify.jl DEVICE CHECKPOINT REFERENCE_JSON [GPU]
#   # DEVICE=amdgpu uses: julia --project=tools/amdgpu tools/verify.jl amdgpu ...
#
# GPU selects the CUDA device (default: 1, the debugging GPU when there are two).
const USAGE = "Usage: verify.jl cpu|metal|cuda|amdgpu CHECKPOINT REFERENCE_JSON [GPU]"

length(ARGS) in (3, 4) || error(USAGE)
const DEVICE = Symbol(ARGS[1])
DEVICE in (:cpu, :metal, :cuda, :amdgpu) || error(USAGE)
DEVICE == :metal && import Metal
DEVICE == :cuda && import CUDA
DEVICE == :amdgpu && import AMDGPU

using JeffClient, Test
import QwenDecisionCore
const JSON = QwenDecisionCore.JSON

matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])

function main(checkpoint, reference, gpu)
    if DEVICE == :cuda
        CUDA.device!(something(gpu, min(1, length(CUDA.devices()) - 1)))
        CUDA.allowscalar(false)
    elseif DEVICE == :metal
        Metal.functional() || error("A functional Apple GPU is required.")
        Metal.allowscalar(false)
    elseif DEVICE == :amdgpu
        AMDGPU.functional() || error("A functional AMD GPU is required.")
        AMDGPU.allowscalar(false)
    end
    backend = NativeBackend(checkpoint; device = DEVICE)
    source = JSON.parsefile(reference)
    cases = source isa AbstractDict ? source["cases"] : source
    @testset "NativeBackend ($DEVICE) versus PyTorch" begin
        for (index, case) in enumerate(cases)
            inputs = Dict(name => matrix(rows, Int64) for (name, rows) in case["inputs"])
            saved = deepcopy(inputs)
            expected = matrix(case["logits"], Float32)
            actual = logits(backend, inputs)
            @test size(actual) == size(expected)
            @test actual ≈ expected atol = 2e-4 rtol = 2e-4
            retained = copy(actual)
            GC.gc(true)
            @test logits(backend, inputs) ≈ expected atol = 2e-4 rtol = 2e-4
            @test actual == retained
            @test inputs == saved
            println(
                get(case, "name", "case $index"),
                ": max logit error ",
                maximum(abs.(actual .- expected)),
            )
            flush(stdout)
        end
    end
end

main(ARGS[2], ARGS[3], length(ARGS) == 4 ? parse(Int, ARGS[4]) : nothing)
