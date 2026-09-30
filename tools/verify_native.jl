using JeffClient
using LinearAlgebra
import JSON

length(ARGS) in (1, 2, 3) || error(
    "Usage: julia --project=tools tools/verify_native.jl CHECKPOINT [cpu|metal] [REFERENCE_JSON]",
)
device = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :cpu
device == :metal && (@eval import Metal)
BLAS.set_num_threads(8)
backend = NativeBackend(ARGS[1]; device)
reference =
    length(ARGS) == 3 ? ARGS[3] :
    joinpath(@__DIR__, "..", "artifacts", "jeff-0.8b-onnx", "reference.json")
samples = JSON.parsefile(reference)
matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
for sample in samples
    inputs = Dict(name => matrix(rows, Int64) for (name, rows) in sample["inputs"])
    expected = matrix(sample["logits"], Float32)
    actual = logits(backend, inputs)
    max_error = maximum(abs.(actual .- expected))
    println("Device: ", device, "; max logit error: ", max_error)
    all(abs.(actual .- expected) .<= 2e-3 .+ 2e-3 .* abs.(expected)) ||
        error("Native Julia scores differ from PyTorch reference.")
end
