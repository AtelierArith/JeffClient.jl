# Reproduces the M2 Max/macOS 26.5.2 abort with Julia 1.13.1 / Metal.jl 1.11.1.
# No JeffClient, Python, checkpoint, or downloads are needed.
# Run: julia --startup-file=no --project=tools tools/mwe_metal_mps_command_buffer.jl
# Control: append --fresh to recreate the MPS wrapper for every encode.
# Expected default: signal 6, _status < MTLCommandBufferStatusCommitted.
using Metal
using Metal: MPS
using Metal.MPSGraphs:
    MPSGraph,
    MPSGraphTensorData,
    placeholderTensor,
    matrixMultiplicationWithPrimaryTensor,
    default_exec_desc
using Metal.ObjectiveC.Foundation: @autoreleasepool, NSDictionary, nil

@autoreleasepool function main(; fresh = false)
    Metal.allowscalar(false)
    a = Metal.ones(Float32, 4096, 4096)
    b = Metal.ones(Float32, 4096, 256)
    c = similar(b)
    graph = MPSGraph()
    pa = placeholderTensor(graph, size(a), Float32)
    pb = placeholderTensor(graph, size(b), Float32)
    result = matrixMultiplicationWithPrimaryTensor(graph, pb, pa)
    feeds = Dict(pa => MPSGraphTensorData(a), pb => MPSGraphTensorData(b))
    results = Dict(result => MPSGraphTensorData(c))
    feed_dictionary, result_dictionary = NSDictionary(feeds), NSDictionary(results)
    descriptor = default_exec_desc()
    Metal.synchronize()
    queue = Metal.global_queue(Metal.device())
    cached_owner = cached_command = nothing
    println("Julia $(VERSION), Metal $(pkgversion(Metal)), fresh=$fresh")
    for i = 1:16
        Metal.end_encoder!(queue)
        owner = Metal.ensure_cmdbuf!(queue)
        command =
            !fresh && cached_owner === owner ? cached_command : MPS.MPSCommandBuffer(owner)
        cached_owner, cached_command = owner, command
        MPS.encode!(command, graph, feed_dictionary, result_dictionary, nil, descriptor)
        # MPSGraph can commit the original buffer and continue on a new one.
        # The Metal queue still holds `owner`; its next compute encoder aborts.
        println("encode $i: queue=$(owner.status), MPS=$(command.commandBuffer.status)")
        flush(stdout)
        Metal.record_operation!(queue, a, b, c, feeds, results, command)
        Metal.maybe_autoflush!(queue)
        c .+= 1.0f0
    end
    Metal.synchronize()
    @assert all(==(4097.0f0), Array(c))
    println("PASS: numerical result verified")
end

all(==("--fresh"), ARGS) || error("Only --fresh is supported.")
main(; fresh = "--fresh" in ARGS)
