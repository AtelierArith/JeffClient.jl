using JeffClient, LinearAlgebra, BenchmarkTools
import JSON
if JeffClient.cpu_setting(:accelerate)
    import AppleAccelerate
end

function measure_projection(name, weight, input, samples)
    output = Matrix{Float32}(undef, size(weight, 2), size(input, 2))
    mul!(output, transpose(weight), input)
    trial =
        @benchmark mul!($output, transpose($weight), $input) samples=samples evals=1 seconds=10
    estimate = median(trial)
    return Dict(
        "name"=>name,
        "weight_shape"=>collect(size(weight)),
        "input_shape"=>collect(size(input)),
        "median_ms"=>estimate.time/1e6,
        "heap_bytes"=>estimate.memory,
        "allocations"=>estimate.allocs,
        "gflops_per_second"=>2.0*size(weight, 1)*size(weight, 2)*size(input, 2)/estimate.time,
    )
end

function projection_sweep!(output, weights, input)
    for weight in weights
        mul!(output, transpose(weight), input)
    end
    return output
end

function measure_sweep(name, weights, input, samples)
    output = Matrix{Float32}(undef, size(first(weights), 2), size(input, 2))
    projection_sweep!(output, weights, input)
    trial =
        @benchmark projection_sweep!($output, $weights, $input) samples=samples evals=1 seconds=10
    estimate = median(trial)
    return Dict(
        "name"=>name,
        "weight_count"=>length(weights),
        "median_sweep_ms"=>estimate.time/1e6,
        "mean_per_weight_ms"=>estimate.time/1e6/length(weights),
        "heap_bytes"=>estimate.memory,
        "allocations"=>estimate.allocs,
    )
end

function main()
    length(ARGS) in 2:4 || error(
        "Usage: benchmark_cpu_projections.jl CHECKPOINT REFERENCE_JSON [SAMPLES] [OUTPUT_JSON]",
    )
    samples = length(ARGS)>=3 ? parse(Int, ARGS[3]) : 50
    samples >= 3 || error("Use at least 3 samples")
    JeffClient.initialize_cpu!()
    backend = NativeBackend(ARGS[1])
    references = JSON.parsefile(ARGS[2])
    sample = first(references isa AbstractDict ? references["cases"] : references)
    ids = Int64.(first(sample["inputs"]["input_ids"]))
    mask = Int64.(first(sample["inputs"]["attention_mask"]))
    start = JeffClient.cpu_setting(:trim_padding) ? findfirst(!iszero, mask) : 1
    hidden = JeffClient.native_gather(backend.embedding, ids[start:end])
    mask = mask[start:end]
    seen = Set{Symbol}()
    results = []
    for layer in backend.layers
        kind = layer.attention.kind
        normalized = JeffClient.native_rms(hidden, layer.input_norm, backend.config.eps)
        mixed =
            kind == :full ?
            JeffClient.full_attention(layer.attention, normalized, mask, backend.config) :
            JeffClient.delta_attention(layer.attention, normalized, mask, backend.config)
        residual, mlp_input = JeffClient.native_residual_rms(
            hidden,
            mixed,
            layer.post_norm,
            backend.config.eps,
        )
        if !(kind in seen)
            attention_input =
                kind == :delta ? normalized .* reshape(Float32.(mask), 1, :) : normalized
            for key in (kind == :full ? (:q, :k, :v) : (:qkv, :z, :a, :b))
                push!(
                    results,
                    measure_projection(
                        "$(kind).$key",
                        getproperty(layer.attention, key),
                        attention_input,
                        samples,
                    ),
                )
            end
            for key in (:gate, :up)
                push!(
                    results,
                    measure_projection(
                        "$(kind).mlp.$key",
                        getproperty(layer.mlp, key),
                        mlp_input,
                        samples,
                    ),
                )
            end
            gated = JeffClient.native_mlp_gate!(
                JeffClient.native_linear(layer.mlp.gate, mlp_input),
                JeffClient.native_linear(layer.mlp.up, mlp_input),
            )
            push!(
                results,
                measure_projection("$(kind).mlp.down", layer.mlp.down, gated, samples),
            )
            push!(seen, kind)
        end
        hidden = JeffClient.native_residual_add!(
            residual,
            JeffClient.native_mlp(layer.mlp, mlp_input),
        )
        length(seen)==2 && break
    end
    sweeps = []
    for key in (:gate, :up, :down)
        weights = [getproperty(layer.mlp, key) for layer in backend.layers]
        input = zeros(Float32, size(first(weights), 1), length(mask))
        push!(sweeps, measure_sweep("mlp.$key", weights, input, samples))
    end
    report = Dict(
        "julia_version"=>string(VERSION),
        "cpu"=>Sys.CPU_NAME,
        "samples"=>samples,
        "blas_config"=>string(BLAS.get_config()),
        "sequence_length"=>length(mask),
        "projections"=>results,
        "weight_sweeps"=>sweeps,
        "scope"=>"First representative full and delta layers; isolated warmed mul!, outputs preallocated; excludes attention and activation",
    )
    println(JSON.json(report, 2))
    length(ARGS)>=4 && (mkpath(dirname(ARGS[4])); write(ARGS[4], JSON.json(report, 2)*"\n"))
end
main()
