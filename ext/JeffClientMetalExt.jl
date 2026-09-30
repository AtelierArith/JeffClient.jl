module JeffClientMetalExt

import JeffClient
import Metal
using LinearAlgebra
using Metal: MPS
using Metal.MPSGraphs:
    MatmulGraphKey,
    MPSGraph,
    MPSGraphTensor,
    MPSGraphTensorData,
    placeholderTensor,
    transposeTensor,
    matrixMultiplicationWithPrimaryTensor,
    default_exec_desc
using Metal.ObjectiveC.Foundation: @autoreleasepool, NSDictionary, nil

include("metal_buffers.jl")

function JeffClient.native_array(::Val{:metal}, x)
    Metal.functional() || throw(ArgumentError("Metal is not available on this machine."))
    return Metal.MtlArray(x)
end

# The product graph has no destination input or beta*C expression: old pooled
# contents are never read, unlike a generic GEMM graph with beta set to zero.
function JeffClient.native_matmul(
    a::Union{Metal.MtlMatrix,Transpose{<:Any,<:Metal.MtlMatrix}},
    b::Union{Metal.MtlMatrix,Transpose{<:Any,<:Metal.MtlMatrix}},
)
    size(a, 2) == size(b, 1) ||
        throw(DimensionMismatch("Matrix inner dimensions must match."))
    result = pooled_array(Float32, (size(a, 1), size(b, 2)))
    left = a isa Transpose ? parent(a) : a
    right = b isa Transpose ? parent(b) : b
    batched_matmul!(
        result,
        left,
        right,
        a isa Transpose ? 'T' : 'N',
        b isa Transpose ? 'T' : 'N',
    )
    return result
end

struct ProductGraph
    graph::MPSGraph
    place_a::MPSGraphTensor
    place_b::MPSGraphTensor
    result::MPSGraphTensor
end

function ProductGraph(key::MatmulGraphKey{T,T}) where {T}
    graph = MPSGraph()
    place_a = placeholderTensor(graph, key.size_a, T)
    place_b = placeholderTensor(graph, key.size_b, T)
    a =
        key.transpose_a == 'T' ?
        transposeTensor(graph, place_a, key.ndims_a - 2, key.ndims_a - 1) : place_a
    b =
        key.transpose_b == 'T' ?
        transposeTensor(graph, place_b, key.ndims_b - 2, key.ndims_b - 1) : place_b
    # MPSGraph tensor shapes reverse Julia's column-major dimensions.
    result = matrixMultiplicationWithPrimaryTensor(graph, b, a)
    return ProductGraph(graph, place_a, place_b, result)
end

const PRODUCT_GRAPH_CACHE = Dict{MatmulGraphKey,ProductGraph}()
const PRODUCT_GRAPH_LOCK = ReentrantLock()

# Following Laya, encode the cached MPSGraph into Metal's current batch instead
# of committing a separate command buffer (and flushing kernels) per product.
# All arrays and Objective-C feed objects stay rooted until GPU completion.
@autoreleasepool function batched_matmul!(c, a, b, transpose_a, transpose_b)
    key = MatmulGraphKey(a, b, c, true, false, transpose_a, transpose_b)
    cached = @lock PRODUCT_GRAPH_LOCK get!(PRODUCT_GRAPH_CACHE, key) do
        ProductGraph(key)
    end
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.place_a => MPSGraphTensorData(a),
        cached.place_b => MPSGraphTensorData(b),
    )
    results =
        Dict{MPSGraphTensor,MPSGraphTensorData}(cached.result => MPSGraphTensorData(c))
    queue = Metal.global_queue(Metal.device())
    Metal.end_encoder!(queue)
    command = MPS.MPSCommandBuffer(Metal.ensure_cmdbuf!(queue))
    MPS.encode!(
        command,
        cached.graph,
        NSDictionary(feeds),
        NSDictionary(results),
        nil,
        default_exec_desc(),
    )
    Metal.record_operation!(queue, a, b, c, feeds, results, command)
    Metal.maybe_autoflush!(queue)
    return c
end

# Solve each right-hand-side column independently, in row order. Forming a
# finite-series inverse with matrix products is unstable for trained DeltaNet
# heads, even though the series terminates exactly in exact arithmetic.
function unit_lower_solve_kernel!(output, system, rhs, n, columns)
    column = Int(Metal.thread_position_in_grid_1d())
    if column <= columns
        for row = 1:n
            value = rhs[row, column]
            for previous = 1:(row-1)
                value -= system[row, previous] * output[previous, column]
            end
            output[row, column] = value
        end
    end
    return
end

function JeffClient.delta_solve(system::Metal.MtlMatrix, rhs::Metal.MtlMatrix)
    n, columns = size(rhs)
    size(system) == (n, n) || throw(
        DimensionMismatch(
            "Expected a square triangular system matching the right-hand side.",
        ),
    )
    output = pooled_array(eltype(rhs), size(rhs))
    threads = min(columns, 256)
    Metal.@metal threads=threads groups=cld(columns, threads) unit_lower_solve_kernel!(
        output,
        system,
        rhs,
        n,
        columns,
    )
    return output
end

include("metal_delta.jl")
include("metal_normalization.jl")
include("metal_softmax.jl")
include("metal_attention.jl")

end
