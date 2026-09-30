using JeffClient
import JSON
import Metal

function main()
    length(ARGS) in (2, 3) || error(
        "Usage: julia --project=tools tools/verify_metal.jl CHECKPOINT REFERENCE_JSON [REPEATS]",
    )
    repeats = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 2
    repeats >= 1 || error("REPEATS must be positive.")
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    println(
        "Julia ",
        VERSION,
        "; Metal ",
        pkgversion(Metal),
        "; device ",
        Metal.device().name,
    )
    flush(stdout)
    report = JSON.parsefile(ARGS[2])
    report["format_version"] == 1 || error("Unsupported reference format.")
    backend = NativeBackend(ARGS[1]; device = :metal)
    rows_to_matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    maximum_error = 0.0f0
    for pass = 1:repeats
        for sample in report["cases"]
            inputs = Dict(
                name => rows_to_matrix(rows, Int64) for (name, rows) in sample["inputs"]
            )
            expected = rows_to_matrix(sample["logits"], Float32)
            actual = logits(backend, inputs)
            all(isfinite, actual) || error("Nonfinite logits: $(sample["name"])")
            size(actual) == size(expected) || error("Logit shapes do not match.")
            logit_error = maximum(abs.(actual .- expected))
            maximum_error = max(maximum_error, logit_error)
            println("Pass ", pass, "; ", sample["name"], "; max logit error ", logit_error)
            flush(stdout)
            all(abs.(actual .- expected) .<= 2.0f-4 .+ 2.0f-4 .* abs.(expected)) ||
                Base.error(
                    "Metal logits differ from independent PyTorch reference: $(sample["name"])",
                )
            # Force finalization between cases, so queued-operation lifetimes and
            # reuse of MPS destinations are exercised as well as a fresh forward.
            GC.gc(true)
        end
    end
    println(
        "Validated ",
        length(report["cases"]),
        " cases × ",
        repeats,
        " passes; max logit error ",
        maximum_error,
    )
end

main()
