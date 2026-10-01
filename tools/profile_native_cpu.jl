using JeffClient
using LinearAlgebra
using InteractiveUtils
using Profile
import JSON
import JET
if get(ENV, "JEFF_CPU_OCTAVIAN_DELTA", "0") == "1"
    import Octavian
end

if get(ENV, "JEFF_CPU_ACCELERATE", "0") == "1"
    Sys.isapple() || error("Apple Accelerate requires macOS.")
    import AppleAccelerate
    any(lib -> occursin("Accelerate", lib.libname), BLAS.get_config().loaded_libs) ||
        error("Accelerate BLAS forwarding requires macOS 13.4 or later.")
end

function main()
    length(ARGS) == 2 || error("Usage: profile_native_cpu.jl CHECKPOINT REFERENCE_JSON")
    BLAS.set_num_threads(8)
    sample = only(JSON.parsefile(ARGS[2]))
    inputs = Dict(
        name => reduce(vcat, [permutedims(Int64.(row)) for row in rows]) for
        (name, rows) in sample["inputs"]
    )
    backend = NativeBackend(ARGS[1])
    println(
        "Julia ",
        VERSION,
        "; BLAS threads: ",
        BLAS.get_num_threads(),
        "; ",
        BLAS.get_config(),
    )
    @code_warntype logits(backend, inputs)
    if JET.JET_AVAILABLE
        extension = Base.get_extension(JeffClient, :JeffClientAppleAccelerateExt)
        modules = extension === nothing ? (JeffClient,) : (JeffClient, extension)
        octavian = Base.get_extension(JeffClient, :JeffClientOctavianExt)
        octavian === nothing || (modules = (modules..., octavian))
        show(
            stdout,
            MIME"text/plain"(),
            JET.@report_opt target_modules=modules logits(backend, inputs)
        )
    end
    logits(backend, inputs)
    GC.gc(true)
    measured = @timed logits(backend, inputs)
    println(
        "\nWarm forward: ",
        measured.time,
        " s; ",
        measured.bytes,
        " bytes; GC ",
        measured.gctime,
        " s",
    )
    Profile.clear()
    Profile.@profile for _ = 1:5
        logits(backend, inputs)
    end
    Profile.print(;
        format = :flat,
        sortedby = :count,
        mincount = 20,
        C = true,
        groupby = :thread,
    )
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=0.05 logits(backend, inputs)
    sites = Dict{String,Tuple{Int,Int}}()
    for record in Profile.Allocs.fetch().allocs
        frame = findfirst(f -> occursin("/src/native", string(f.file)), record.stacktrace)
        frame === nothing && continue
        site = string(record.stacktrace[frame])
        count, bytes = get(sites, site, (0, 0))
        sites[site] = (count + 1, bytes + record.size)
    end
    println("\nSampled allocation sites (5%, not total bytes):")
    for (site, (count, bytes)) in
        sort(collect(sites); by = p -> last(p)[2], rev = true)[1:min(20, length(sites))]
        println(bytes, " bytes / ", count, " samples: ", site)
    end
end

main()
