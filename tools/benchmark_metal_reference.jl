# Equal-work GPU diagnostic. All positions reach the final residual/RMS.
using JeffClient
using QwenDecisionCore
import Metal

Metal.functional() || error("A functional Apple GPU is required.")
Metal.allowscalar(false)
for flag in (
    "QDC_METAL_BATCHED",
    "QDC_METAL_WORKSPACE",
    "QDC_METAL_PACKED_MLP",
    "QDC_METAL_TRIM_PADDING",
    "QDC_METAL_SHAPE_WORKSPACES",
    "QDC_METAL_FUSED_DELTA_MASK",
)
    ENV[flag] = "0"
end
extension = Base.get_extension(QwenDecisionCore, :QwenDecisionCoreMetalExt)
extension === nothing && error("QwenDecisionCoreMetalExt was not loaded.")
source_path = joinpath(pkgdir(QwenDecisionCore), "ext", "metal_normalization.jl")
source = read(source_path, String)
start = findfirst("function QwenDecisionCore.native_hidden_forward(", source)
stop = findnext("function QwenDecisionCore.native_rms(", source, last(start))
method_source = source[first(start):prevind(source, first(stop))]
tail = findfirst("    # The last layer needs only the readout column", method_source)
tail === nothing && error("Metal reference adapter no longer matches the source.")
method_source = method_source[1:prevind(method_source, first(tail))] * """
    _, normalized = residual_input_rms!(residual, mlp, final_norm, cfg.eps)
    return normalized[:, end:end]
end
"""
# Only this benchmark process changes the method. No package source is modified.
include_string(extension, method_source, source_path)
if !isempty(ARGS) && first(ARGS) == "--verify"
    popfirst!(ARGS)
    include(joinpath(@__DIR__, "verify_metal.jl"))
else
    source = read(joinpath(@__DIR__, "benchmark_inference.jl"), String)
    source = replace(source, r"main\(\)\s*$" => "")
    source = replace(
        source,
        "samples=samples evals=1 seconds=120" => "samples=samples evals=1 seconds=120 gctrial=false gcsample=false",
        "\"samples\" => length(trial.times)," => "\"times_ms\" => trial.times ./ 1e6, \"samples\" => length(trial.times),",
    )
    include_string(Main, source, joinpath(@__DIR__, "benchmark_inference.jl"))
    Base.invokelatest() do
        run_benchmark(ARGS, :metal_full_sequence_reference)
    end
end
