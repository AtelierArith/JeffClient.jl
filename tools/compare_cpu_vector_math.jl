using JeffClient, LoopVectorization, BenchmarkTools
@assert Base.get_extension(JeffClient, :JeffClientLoopVectorizationExt) !== nothing

# Algebraic experiment inspired by NNlib.sigmoid_fast. Not a replacement for
# testing NNlib itself, nor a production activation: its saturation semantics
# differ from JeffClient's scalar formula at extreme negative inputs.
@inline function nnlib_style_silu(x::Float32)
    t = @fastmath exp(-abs(x))
    y = ifelse(x >= 0, inv(1.0f0 + t), t / (1.0f0 + t))
    return x * ifelse(x > 40, one(y), ifelse(x < -80, zero(y), y))
end

function main()
    for width in (3584, 6144)
        input = reshape(0.1f0 .+ sin.(Float32.(1:(width*101))), width, 101)
        for enabled in ("0", "1")
            JeffClient.with_cpu_settings(:portable_vector_math => enabled) do
                trial =
                    @benchmark JeffClient.native_cpu_silu!(x) setup=(x=copy($input)) evals=1 samples=50
                estimate = median(trial)
                println(
                    "width=",
                    width,
                    " vector=",
                    enabled,
                    " median_us=",
                    estimate.time/1000,
                    " bytes=",
                    estimate.memory,
                )
            end
        end
        trial =
            @benchmark (x .= nnlib_style_silu.(x)) setup=(x=copy($input)) evals=1 samples=50
        println("width=", width, " nnlib_style=true median_us=", median(trial).time/1000)
    end
end
main()
