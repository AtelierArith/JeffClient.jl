using JeffClient, LinearAlgebra
import JSON
if JeffClient.cpu_setting(:accelerate)
    import AppleAccelerate
end
BLAS.set_num_threads(8)
function validate()
    length(ARGS) == 2 || error("Usage: validate_native_cpu.jl CHECKPOINT REFERENCE_JSON")
    checkpoint, reference = ARGS
    source = JSON.parsefile(reference)
    cases = source isa AbstractDict ? source["cases"] : source
    backend = NativeBackend(checkpoint)
    matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    max_error = 0.0f0
    try
        for index in vcat(collect(eachindex(cases)), [1, 15, 2, 1])
            sample = cases[index]
            inputs = Dict(name => matrix(rows, Int64) for (name, rows) in sample["inputs"])
            original = deepcopy(inputs)
            expected = matrix(sample["logits"], Float32)
            retained = nothing
            saved = nothing
            for iteration = 1:2
                iteration == 2 && GC.gc(true)
                actual = logits(backend, inputs)
                delta = abs.(actual .- expected)
                all(isfinite, actual) || error("nonfinite case $index")
                all(delta .<= 2.0f-4 .+ 2.0f-4 .* abs.(expected)) ||
                    error("reference mismatch case $index")
                inputs == original || error("inputs modified case $index")
                max_error = max(max_error, maximum(delta))
                if iteration == 1
                    retained = actual
                    saved = copy(actual)
                else
                    retained == saved || error("prior scores modified case $index")
                end
            end
            println("passed case ", index, " shape ", size(inputs["input_ids"]))
            flush(stdout)
        end
        println("max_logit_error=", max_error)
    finally
        nothing # NativeBackend owns ordinary Julia arrays; no close method.
    end
end
validate()
