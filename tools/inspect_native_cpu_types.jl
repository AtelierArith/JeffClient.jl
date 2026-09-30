using JeffClient, InteractiveUtils
import Cthulhu, TypedSyntax, JSON
if get(ENV, "JEFF_CPU_ACCELERATE", "0") == "1"
    import AppleAccelerate
end

function main()
    length(ARGS) in 2:4 || error(
        "Usage: inspect_native_cpu_types.jl CHECKPOINT REFERENCE_JSON [all|hidden|workspace|layer|mlp|gate] [typed|source|descend]",
    )
    target = length(ARGS) >= 3 ? ARGS[3] : "all"
    mode = length(ARGS) >= 4 ? ARGS[4] : "typed"
    mode in ("typed", "source", "descend") || error("Unknown mode")
    mode == "descend" && !(stdin isa Base.TTY) && error("Descent requires a terminal")
    # Source annotations may expand Core.Const(ENV). Keep this diagnostic
    # process from printing unrelated credentials or other environment data.
    if mode in ("source", "descend")
        allowed = Set([
            "HOME",
            "PATH",
            "TMPDIR",
            "TERM",
            "LANG",
            "JULIA_DEPOT_PATH",
            "JULIA_REVISE_POLL",
        ])
        for key in collect(keys(ENV))
            (key in allowed || startswith(key, "JEFF_CPU_")) || delete!(ENV, key)
        end
    end
    backend = NativeBackend(ARGS[1])
    reference = JSON.parsefile(ARGS[2])
    sample = first(reference isa AbstractDict ? reference["cases"] : reference)
    ids = Int64.(first(sample["inputs"]["input_ids"]))
    mask = Int64.(first(sample["inputs"]["attention_mask"]))
    start = findfirst(!iszero, mask)
    hidden = JeffClient.native_gather(backend.embedding, ids[start:end])
    mask = mask[start:end]
    layer = first(backend.layers)
    cfg = backend.config
    width = size(layer.mlp.gate, 2)
    buffers =
        (zeros(Float32, width, size(hidden, 2)), zeros(Float32, width, size(hidden, 2)))
    targets = (
        hidden = (
            JeffClient.native_hidden_forward,
            (hidden, backend.layers, mask, backend.final_norm, cfg),
        ),
        workspace = (JeffClient.cpu_mlp_workspace, (backend.layers, size(hidden, 2))),
        layer = (
            JeffClient.cpu_layer_with_mlp_workspace,
            (layer, hidden, mask, cfg, buffers),
        ),
        mlp = (JeffClient.native_mlp, (layer.mlp, hidden, buffers)),
        gate = (JeffClient.cpu_owned_mlp_gate!, buffers),
    )
    println(
        "Julia ",
        VERSION,
        "; Cthulhu ",
        pkgversion(Cthulhu),
        "; TypedSyntax ",
        pkgversion(TypedSyntax),
    )
    for name in (target == "all" ? keys(targets) : (Symbol(target),))
        f, args = getproperty(targets, name)
        types = Tuple{map(typeof, args)...}
        println(
            "\n=== ",
            name,
            " ===\nMethod: ",
            which(f, types),
            "\nReturn types: ",
            Base.return_types(f, types),
        )
        if mode == "descend"
            Cthulhu.descend(f, types)
        elseif mode == "source"
            node = TypedSyntax.TypedSyntaxNode(f, types; strip_macros = true)
            node === nothing ? code_warntype(stdout, f, types; optimize = false) :
            printstyled(stdout, node; hide_type_stable = false, iswarn = true)
        else
            code_warntype(stdout, f, types; optimize = false)
        end
    end
end
main()
