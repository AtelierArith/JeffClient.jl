# Regression for MPSGraph internally committing a reused MPSCommandBuffer.
# Run: julia --startup-file=no --project=tools tools/verify_metal_command_buffers.jl
using JeffClient
using QwenDecisionCore
import Metal

function verify_metal_command_buffers()
    Metal.functional() || error("A functional Apple GPU is required.")
    Metal.allowscalar(false)
    a = Metal.ones(Float32, 4096, 4096)
    previous = nothing
    for (tokens, increment, expected) in
        ((256, 1.0f0, 4097.0f0), (9, 2.0f0, 4098.0f0), (256, 3.0f0, 4099.0f0))
        b = Metal.ones(Float32, 4096, tokens)
        outputs = Metal.MtlMatrix{Float32}[]
        for _ = 1:16
            c = QwenDecisionCore.native_matmul(a, b)
            c .+= increment
            push!(outputs, c)
        end
        GC.gc(true) # Wrappers and tensor-data must survive queued GPU work.
        Metal.synchronize()
        for c in outputs
            all(==(expected), Array(c)) || error("Queued matrix product changed.")
        end
        all(==(1.0f0), Array(b)) || error("Matrix product changed its input.")
        if previous !== nothing
            all(==(previous[2]), Array(previous[1])) || error("Retained output changed.")
        end
        previous = (first(outputs), expected)
        GC.gc(true)
        println(
            "Validated 16 queued matrix products; tokens=$tokens; GC and retained output.",
        )
        flush(stdout)
    end
    all(==(1.0f0), Array(a)) || error("Matrix product changed its weight.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_metal_command_buffers()
end
