using JeffClient
using InteractiveUtils
using LinearAlgebra
using Profile
import JSON
import JET
import Metal

function profile_forwards(backend, inputs, iterations)
    for _ = 1:iterations
        logits(backend, inputs)
    end
    return nothing
end

function report_target(name, f, args...)
    println("\n=== ", name, " ===")
    println("Argument types: ", typeof(args))
    println("Return inference: ", Base.return_types(f, Tuple{map(typeof, args)...}))
    @code_warntype f(args...)
    if JET.JET_AVAILABLE
        jet = JET.@report_opt target_modules=(
            JeffClient,
            Base.get_extension(JeffClient, :JeffClientMetalExt),
        ) f(args...)
        show(stdout, MIME"text/plain"(), jet)
        println()
    end
    flush(stdout)
end

function main()
    length(ARGS) == 2 || error(
        "Usage: julia --project=tools tools/inspect_native.jl CHECKPOINT REFERENCE_JSON",
    )
    BLAS.set_num_threads(8)
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    backend = NativeBackend(ARGS[1]; device = :metal)
    sample = first(JSON.parsefile(ARGS[2]))
    rows_to_matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    inputs = Dict(name => rows_to_matrix(rows, Int64) for (name, rows) in sample["inputs"])
    println(
        "Julia ",
        VERSION,
        "; JET ",
        pkgversion(JET),
        "; JET available: ",
        JET.JET_AVAILABLE,
    )
    println("Backend type: ", typeof(backend))
    println(
        "Layer storage type: ",
        typeof(backend.layers),
        "; eltype: ",
        eltype(backend.layers),
    )
    report_target("logits", logits, backend, inputs)
    hidden = JeffClient.native_gather(backend.embedding, vec(inputs["input_ids"][1, :]))
    mask = vec(inputs["attention_mask"][1, :])
    delta = first(layer for layer in backend.layers if layer.attention.kind == :delta)
    full = first(layer for layer in backend.layers if layer.attention.kind == :full)
    report_target(
        "delta layer",
        JeffClient.native_layer,
        delta,
        hidden,
        mask,
        backend.config,
    )
    report_target("full layer", JeffClient.native_layer, full, hidden, mask, backend.config)
    report_target(
        "DeltaNet",
        JeffClient.delta_attention,
        delta.attention,
        hidden,
        mask,
        backend.config,
    )
    report_target(
        "linear projection",
        JeffClient.native_linear,
        delta.attention.qkv,
        hidden,
    )
    report_target(
        "Metal matmul",
        JeffClient.native_matmul,
        transpose(delta.attention.qkv),
        hidden,
    )
    get(ENV, "JEFF_INSPECT_PROFILE", "1") == "0" && return
    println("\n=== Warm allocation profile ===")
    logits(backend, inputs)
    Metal.synchronize()
    GC.gc(true)
    measured = @timed logits(backend, inputs)
    println(
        "Warm forward: seconds=",
        measured.time,
        "; Julia bytes=",
        measured.bytes,
        "; GC seconds=",
        measured.gctime,
    )
    println("\n=== Warm CPU sampling profile (GPU submission and waiting included) ===")
    iterations = parse(Int, get(ENV, "JEFF_PROFILE_ITERATIONS", "20"))
    iterations > 0 || error("JEFF_PROFILE_ITERATIONS must be positive.")
    println("Profiled warmed forwards: ", iterations)
    profile_forwards(backend, inputs, 3)
    Metal.synchronize()
    Profile.init(; delay = 0.001)
    Profile.clear()
    Profile.@profile profile_forwards(backend, inputs, iterations)
    Metal.synchronize()
    Profile.print(;
        format = :flat,
        sortedby = :count,
        mincount = 10,
        C = true,
        groupby = :thread,
    )
    println("\n=== CPU sampling call tree ===")
    Profile.print(; format = :tree, maxdepth = 18, mincount = 10, C = false)
    println("\n=== Sampled allocation stacks ===")
    sample_rate = parse(Float64, get(ENV, "JEFF_ALLOC_SAMPLE_RATE", "0.01"))
    0 < sample_rate <= 1 || error("JEFF_ALLOC_SAMPLE_RATE must be in (0, 1].")
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=sample_rate logits(backend, inputs)
    Metal.synchronize()
    records = Profile.Allocs.fetch().allocs
    by_site = Dict{String,Tuple{Int,Int}}()
    by_type = Dict{String,Tuple{Int,Int}}()
    for allocation in records
        relevant = findfirst(
            frame -> occursin("/JeffClient.jl/", string(frame.file)),
            allocation.stacktrace,
        )
        site = if relevant === nothing
            "outside JeffClient"
        else
            frame = allocation.stacktrace[relevant]
            "$(basename(string(frame.file))):$(frame.line) $(frame.func)"
        end
        count, bytes = get(by_site, site, (0, 0))
        by_site[site] = (count + 1, bytes + allocation.size)
        key = string(allocation.type)
        count, bytes = get(by_type, key, (0, 0))
        by_type[key] = (count + 1, bytes + allocation.size)
    end
    println(
        "Sampled allocations: ",
        length(records),
        " (sample_rate=",
        sample_rate,
        "; sizes below are sampled, not totals)",
    )
    for (label, groups) in (("Sites", by_site), ("Types", by_type))
        println(label, " sorted by sampled allocation count:")
        for (key, value) in first(
            sort!(collect(groups); by = x -> last(x)[1], rev = true),
            min(15, length(groups)),
        )
            println(value, " ", key)
        end
    end
end

main()
