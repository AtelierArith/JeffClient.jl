using QwenDecisionCore
# Isolated diagnostic wrapper; production CPU defaults are unchanged.
source = read(joinpath(@__DIR__, "benchmark_inference.jl"), String)
source = replace(source, r"main\(\)\s*$" => "")
source = replace(
    source,
    "samples=samples evals=1 seconds=120" => "samples=samples evals=1 seconds=120 gctrial=false gcsample=false",
    "\"samples\" => length(trial.times)," => "\"times_ms\" => trial.times ./ 1e6, \"samples\" => length(trial.times),",
)
include_string(Main, source, abspath(joinpath(@__DIR__, "benchmark_inference.jl")))
QwenDecisionCore.initialize_cpu!()
BLAS.set_num_threads(1)
AppleAccelerate.set_num_threads(1)
@assert AppleAccelerate.get_num_threads() == 1
QwenDecisionCore.with_cpu_settings(
    :trim_padding => false,
    :final_query => false,
    :final_token_only => false,
    :recurrent_delta => false,
    :octavian_delta => false,
    :delta_chunk_size => 64,
) do
    run_benchmark(ARGS, :python_reference_single_thread)
end
