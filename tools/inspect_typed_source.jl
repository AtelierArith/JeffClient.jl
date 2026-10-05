using JeffClient
using QwenDecisionCore
using InteractiveUtils
import Cthulhu
import JSON
import Metal
import TypedSyntax

const INSPECTION_TARGETS =
    ("logits", "full", "delta", "rope", "submit", "tensor", "cached-tensor")

function inspection_target(target, backend, inputs)
    target == "logits" && return logits, (backend, inputs)
    extension = Base.get_extension(QwenDecisionCore, :QwenDecisionCoreMetalExt)
    hidden = QwenDecisionCore.native_gather(
        backend.backbone.embedding,
        vec(inputs["input_ids"][1, :]),
    )
    mask = vec(inputs["attention_mask"][1, :])
    cfg = backend.backbone.config
    full =
        first(layer for layer in backend.backbone.layers if layer.attention.kind == :full)
    delta =
        first(layer for layer in backend.backbone.layers if layer.attention.kind == :delta)
    target == "full" &&
        return QwenDecisionCore.full_attention, (full.attention, hidden, mask, cfg)
    target == "delta" &&
        return QwenDecisionCore.delta_attention, (delta.attention, hidden, mask, cfg)
    target == "rope" && return extension.rope_tables, (hidden, cfg, size(hidden, 2))
    if target == "submit"
        weight = delta.attention.qkv
        output = extension.pooled_array(Float32, (size(weight, 2), size(hidden, 2)))
        return extension.batched_matmul!, (output, weight, hidden, 'T', 'N')
    end
    target == "tensor" && return extension.MPSGraphTensorData, (hidden,)
    if target == "cached-tensor"
        shape = convert(extension.MPS.MPSShape, reverse(size(hidden)))
        return extension.graph_tensor_data, (hidden, shape)
    end
    error("Unknown inspection target: $target")
end

function inspect_source(target, f, args, mode)
    types = Tuple{map(typeof, args)...}
    println("\n=== ", target, " ===")
    println("Method: ", which(f, types))
    println("Argument types: ", types)
    println("Return inference: ", Base.return_types(f, types))
    if mode == "descend"
        Cthulhu.descend(f, types)
    elseif mode == "typed"
        code_warntype(stdout, f, types; optimize = false)
    else
        node = TypedSyntax.TypedSyntaxNode(f, types; strip_macros = true)
        if node === nothing
            println("Source mapping unavailable; showing inferred IR.")
            code_warntype(stdout, f, types; optimize = false)
        else
            printstyled(stdout, node; hide_type_stable = false, iswarn = false)
            println()
        end
    end
    flush(stdout)
    return nothing
end

function main()
    length(ARGS) in 2:4 || error(
        "Usage: julia --project=tools tools/inspect_typed_source.jl CHECKPOINT REFERENCE_JSON [all|logits|full|delta|rope|submit|tensor|cached-tensor] [source|typed|descend]",
    )
    target = length(ARGS) >= 3 ? ARGS[3] : "all"
    mode = length(ARGS) >= 4 ? ARGS[4] : "source"
    target in ("all", INSPECTION_TARGETS...) || error("Unknown target: $target")
    mode in ("source", "typed", "descend") || error("Unknown mode: $mode")
    # Interactive descent requires an explicit choice and a terminal, so batch
    # diagnostics never hang waiting for menu input.
    mode == "descend" &&
        !(stdin isa Base.TTY) &&
        error("Cthulhu descent requires a terminal; use source or typed for batch reports.")
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    println(
        "Julia ",
        VERSION,
        "; Cthulhu ",
        pkgversion(Cthulhu),
        "; TypedSyntax ",
        pkgversion(TypedSyntax),
    )
    println(
        "Source annotations can omit macro/closure details; use typed IR or descend to investigate them.",
    )
    flush(stdout)
    references = JSON.parsefile(ARGS[2])
    cases = references isa AbstractDict ? references["cases"] : references
    sample = first(cases)
    rows_to_matrix(rows, T) = reduce(vcat, [permutedims(T.(row)) for row in rows])
    inputs = Dict(name => rows_to_matrix(rows, Int64) for (name, rows) in sample["inputs"])
    backend = NativeBackend(ARGS[1]; device = :metal)
    for selected in (target == "all" ? INSPECTION_TARGETS : (target,))
        f, args = inspection_target(selected, backend, inputs)
        inspect_source(selected, f, args, mode)
    end
    Metal.synchronize()
    return nothing
end

main()
