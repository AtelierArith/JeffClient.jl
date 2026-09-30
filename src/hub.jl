# The cache strategy follows Laya.jl/src/agent.jl: consult the HF cache, then
# keep complete downloads in package-specific Scratch space.
const CHECKPOINT_REQUIRED = (
    "config.json",
    "decision_config.json",
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
const CHECKPOINT_AUXILIARY = (
    "chat_template.jinja",
    "processor_config.json",
    "model.safetensors.index.json",
    "LICENSE",
    "NOTICE",
)

function safe_relative_path(path::AbstractString)
    isempty(path) && throw(ArgumentError("Path must not be empty."))
    (
        isabspath(path) ||
        occursin('\\', path) ||
        occursin('\0', path) ||
        any(part -> part in ("", ".", ".."), split(path, '/'))
    ) && throw(ArgumentError("Expected a safe relative path: $path"))
    return path
end

function checkpoint_file(path)
    path in CHECKPOINT_REQUIRED ||
        path in CHECKPOINT_AUXILIARY ||
        occursin(r"^model(?:-[0-9]+-of-[0-9]+)?\.safetensors$", path)
end

function complete_checkpoint(dir)
    all(file -> isfile(joinpath(dir, file)), CHECKPOINT_REQUIRED) || return false
    isfile(joinpath(dir, "model.safetensors")) && return true
    index = joinpath(dir, "model.safetensors.index.json")
    isfile(index) || return false
    mapping = JSON.parsefile(index)["weight_map"]
    isempty(mapping) && return false
    return all(values(mapping)) do file
        safe_relative_path(file)
        isfile(joinpath(dir, file))
    end
end

function huggingface_cache()
    haskey(ENV, "HF_HUB_CACHE") && return ENV["HF_HUB_CACHE"]
    haskey(ENV, "HF_HOME") && return joinpath(ENV["HF_HOME"], "hub")
    return joinpath(homedir(), ".cache", "huggingface", "hub")
end

hub_offline() = lowercase(get(ENV, "HF_HUB_OFFLINE", "0")) in ("1", "true", "yes", "on")
hub_root(cache, repo) = joinpath(cache, "models--" * replace(repo, "/" => "--"))

function cached_checkpoint(cache, repo, revision)
    root = hub_root(cache, repo)
    ref = joinpath(root, "refs", revision)
    commit = isfile(ref) ? strip(read(ref, String)) : revision
    occursin(r"^[0-9a-f]{40}$", commit) || return nothing
    dir = joinpath(root, "snapshots", commit)
    return complete_checkpoint(dir) ? dir : nothing
end

function escape_hub_path(path)
    return join(
        split(path, '/') .|>
        part -> join(
            (
                byte in UInt8('a'):UInt8('z') ||
                byte in UInt8('A'):UInt8('Z') ||
                byte in UInt8('0'):UInt8('9') ||
                byte in codeunits("-._~")
            ) ? string(Char(byte)) : "%" * uppercase(string(byte; base = 16, pad = 2))
            for byte in codeunits(part)
        ),
        '/',
    )
end

function publish_file(source, destination)
    mkpath(dirname(destination))
    # rename within the cache filesystem publishes complete files atomically.
    Base.Filesystem.rename(source, destination)
end

"""
    resolve_checkpoint(model_id_or_path; revision="main", cache_dir=nothing,
                       offline=ENV["HF_HUB_OFFLINE"])

Return a complete local Jeff Qwen checkpoint directory. Look up repository IDs
in the existing Hugging Face cache, then in JeffClient's Scratch cache; download
only missing model/config/tokenizer/license files if needed. A local directory
is validated and returned without downloading.

Pass an immutable 40-character commit as `revision` for reproducibility. Branch
names resolve to immutable snapshots on first download; subsequent calls reuse
the cached reference. `cache_dir` overrides Scratch storage for this call.
`HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`, and `HF_HUB_OFFLINE` are
respected. Partial downloads are kept outside completed snapshot files and are
removed on failure. This retrieves source weights, not an ONNX export.
"""
function resolve_checkpoint(
    model_id_or_path::AbstractString;
    revision::AbstractString = "main",
    cache_dir::Union{Nothing,AbstractString} = nothing,
    offline::Bool = hub_offline(),
)
    local_path = expanduser(model_id_or_path)
    if isdir(local_path)
        complete_checkpoint(local_path) ||
            throw(ArgumentError("Incomplete Jeff Qwen checkpoint: $local_path"))
        return abspath(local_path)
    end
    occursin(
        r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$",
        model_id_or_path,
    ) || throw(
        ArgumentError(
            "Use an existing checkpoint directory or an org/model repository ID.",
        ),
    )
    safe_relative_path(revision)
    repo = String(model_id_or_path)
    shared = cached_checkpoint(huggingface_cache(), repo, revision)
    shared === nothing || return shared
    cache =
        cache_dir === nothing ? Scratch.@get_scratch!("hub") :
        abspath(expanduser(cache_dir))
    cached = cached_checkpoint(cache, repo, revision)
    cached === nothing || return cached
    offline && throw(
        ArgumentError("$repo@$revision is not cached; offline mode prevents downloading."),
    )

    endpoint = rstrip(get(ENV, "HF_ENDPOINT", "https://huggingface.co"), '/')
    headers =
        haskey(ENV, "HF_TOKEN") ? ["Authorization" => "Bearer $(ENV["HF_TOKEN"])"] :
        Pair{String,String}[]
    url = "$endpoint/api/models/$repo/revision/$(escape_hub_path(revision))"
    info = JSON.parse(String(take!(Downloads.download(url, IOBuffer(); headers))))
    commit = String(info["sha"])
    occursin(r"^[0-9a-f]{40}$", commit) ||
        throw(ArgumentError("Hub returned an invalid commit ID."))
    occursin(r"^[0-9a-f]{40}$", revision) &&
        revision != commit &&
        throw(ArgumentError("Hub returned a different commit than the requested revision."))
    files = String[
        item["rfilename"] for item in info["siblings"] if checkpoint_file(item["rfilename"])
    ]
    all(file -> file in files, CHECKPOINT_REQUIRED) || throw(
        ArgumentError("$repo@$commit is missing required Jeff Qwen checkpoint files."),
    )
    ("model.safetensors" in files || "model.safetensors.index.json" in files) ||
        throw(ArgumentError("$repo@$commit has no model weights."))
    root = hub_root(cache, repo)
    snapshot = joinpath(root, "snapshots", commit)
    mkpath(root)
    mktempdir(root; prefix = "download-") do temporary
        for file in files
            safe_relative_path(file)
            destination = joinpath(snapshot, file)
            isfile(destination) && continue
            staged = joinpath(temporary, file)
            @info "Downloading Jeff checkpoint file" repo commit file
            Downloads.download(
                "$endpoint/$repo/resolve/$commit/$(escape_hub_path(file))",
                staged;
                headers,
            )
            publish_file(staged, destination)
        end
        complete_checkpoint(snapshot) ||
            throw(ArgumentError("Downloaded checkpoint is incomplete: $snapshot"))
        staged_ref = joinpath(temporary, "revision")
        write(staged_ref, commit)
        publish_file(staged_ref, joinpath(root, "refs", revision))
    end
    return snapshot
end

"""
    load_export(directory; execution_provider=:cpu, provider_options=(;))

Load a bundle produced by tools/export_onnx.jl. Read temperature and the trained
option limit from decision_config.json instead of using uncalibrated defaults.
This loads the prepared-tensor backend; text preparation is a separate step.
"""
function load_export(
    directory::AbstractString;
    execution_provider::Symbol = :cpu,
    provider_options::NamedTuple = (;),
)
    metadata = JSON.parsefile(joinpath(directory, "export_config.json"))
    config = JSON.parsefile(joinpath(directory, "decision_config.json"))
    metadata["format_version"] == 1 || throw(ArgumentError("Unsupported export format."))
    config["format_version"] == 1 ||
        throw(ArgumentError("Unsupported decision checkpoint format."))
    model_file = safe_relative_path(metadata["model_file"])
    return ONNXBackend(
        joinpath(directory, model_file);
        execution_provider,
        provider_options,
        output_name = metadata["output_name"],
        temperature = config["temperature"],
        max_options = config["max_options"],
    )
end
