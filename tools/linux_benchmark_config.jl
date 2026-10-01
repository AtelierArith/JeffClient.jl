using JSON
using SHA
using TOML

function expand_cpus(value)
    cpus = Int[]
    for item in split(value, ',')
        bounds = parse.(Int, split(strip(item), '-'))
        if length(bounds) == 1
            push!(cpus, only(bounds))
        elseif length(bounds) == 2 && bounds[1] <= bounds[2]
            append!(cpus, bounds[1]:bounds[2])
        else
            error("Invalid CPU list: $value")
        end
    end
    length(unique(cpus)) == length(cpus) || error("Duplicate CPU indices.")
    return sort(cpus)
end

function prepare_config(args)
    length(args) == 5 ||
        error("Usage: linux_benchmark_config.jl OUTPUT THREADS CPUS CHECKPOINT REFERENCE")
    output, threads, requested, checkpoint, reference = args
    budget = parse(Int, threads)
    allowed = expand_cpus(
        match(r"Cpus_allowed_list:\s*([^\n]+)", read("/proc/self/status", String))[1],
    )
    topology(cpu) = Tuple(
        strip(read("/sys/devices/system/cpu/cpu$cpu/topology/$name", String)) for
        name in ("physical_package_id", "core_id")
    )
    physical = Dict{Tuple{String,String},Int}()
    for cpu in allowed
        get!(physical, topology(cpu), cpu)
    end
    candidates = sort(collect(values(physical)))
    budget in 1:length(candidates) ||
        error("Requested $budget physical cores; only $(length(candidates)) are allowed.")
    selected = requested == "auto" ? candidates[1:budget] : expand_cpus(requested)
    length(selected) == budget || error("--cpus must select exactly --threads CPUs.")
    all(in(allowed), selected) || error("CPU list exceeds the process's allowed affinity.")
    length(unique(topology.(selected))) == budget ||
        error("Select distinct physical cores, not SMT siblings.")
    root = abspath(joinpath(@__DIR__, ".."))
    source_files = String[]
    for directory in ("src", "ext", "tools")
        for (path, _, names) in walkdir(joinpath(root, directory))
            append!(
                source_files,
                [
                    joinpath(path, name) for
                    name in names if endswith(name, ".jl") || name == "linux-cpu.sh"
                ],
            )
        end
    end
    model_files = [
        joinpath(checkpoint, name) for name in readdir(checkpoint) if
        endswith(name, ".safetensors") || endswith(name, ".json")
    ]
    hashes(files, base) =
        Dict(relpath(file, base) => bytes2hex(open(sha256, file)) for file in files)
    config = Dict(
        "threads" => budget,
        "cpus" => selected,
        "affinity" => join(selected, ','),
        "physical_cores" => topology.(selected),
        "checkpoint" => realpath(checkpoint),
        "model_hashes" => hashes(model_files, checkpoint),
        "reference" => abspath(reference),
        "reference_sha256" => bytes2hex(open(sha256, reference)),
        "source_hashes" => hashes(source_files, root),
        "source_commit" => strip(read(`git -C $root rev-parse HEAD`, String)),
        "source_status" => read(`git -C $root status --short`, String),
        "julia_version" => string(VERSION),
        "machine" => Sys.MACHINE,
        "cpu" => Sys.cpu_info()[1].model,
    )
    manifests = Dict("tools" => joinpath(root, "Manifest.toml"))
    if haskey(ENV, "JEFF_BENCH_BLAS_PROJECT")
        manifests["optional_blas"] =
            joinpath(ENV["JEFF_BENCH_BLAS_PROJECT"], "Manifest.toml")
    end
    config["dependency_versions"] = Dict(
        label => Dict(
            name => get(first(entries), "version", "local") for
            (name, entries) in TOML.parsefile(file)["deps"] if name in (
                "JeffClient",
                "LoopVectorization",
                "BenchmarkTools",
                "PythonCall",
                "OpenBLAS_jll",
                "MKL",
                "MKL_jll",
                "libblastrampoline_jll",
            )
        ) for (label, file) in manifests
    )
    config["manifest_hashes"] =
        Dict(label => bytes2hex(open(sha256, file)) for (label, file) in manifests)
    write(joinpath(output, "config.json"), JSON.json(config, 2) * "\n")
    print(config["affinity"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    prepare_config(ARGS)
end
