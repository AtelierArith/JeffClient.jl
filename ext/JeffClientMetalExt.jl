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
using Metal.ObjectiveC.Foundation: @autoreleasepool, NSDictionary, nil, retain, release

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

mutable struct ProductGraph
    graph::MPSGraph
    place_a::MPSGraphTensor
    place_b::MPSGraphTensor
    result::MPSGraphTensor
    shape_a::MPS.MPSShape
    shape_b::MPS.MPSShape
    shape_c::MPS.MPSShape
end

function ProductGraph(key::MatmulGraphKey{T,T}) where {T}
    graph = MPSGraph()
    shape_a = convert(MPS.MPSShape, reverse(key.size_a))
    shape_b = convert(MPS.MPSShape, reverse(key.size_b))
    shape_c = convert(MPS.MPSShape, reverse(key.size_c))
    place_a = placeholderTensor(graph, shape_a, T)
    place_b = placeholderTensor(graph, shape_b, T)
    a =
        key.transpose_a == 'T' ?
        transposeTensor(graph, place_a, key.ndims_a - 2, key.ndims_a - 1) : place_a
    b =
        key.transpose_b == 'T' ?
        transposeTensor(graph, place_b, key.ndims_b - 2, key.ndims_b - 1) : place_b
    # MPSGraph tensor shapes reverse Julia's column-major dimensions.
    result = matrixMultiplicationWithPrimaryTensor(graph, b, a)
    cached = ProductGraph(graph, place_a, place_b, result, shape_a, shape_b, shape_c)
    # NSArray is an unmanaged autoreleased wrapper in ObjectiveC.jl. Julia's
    # cache alone cannot keep its underlying object alive beyond this pool.
    retain(shape_a)
    retain(shape_b)
    retain(shape_c)
    finalizer(cached) do owner
        release(owner.shape_a)
        release(owner.shape_b)
        release(owner.shape_c)
    end
    return cached
end

const PRODUCT_GRAPH_CACHE = Dict{MatmulGraphKey,ProductGraph}()
const PRODUCT_GRAPH_LOCK = ReentrantLock()

graph_tensor_data(matrix::Metal.MtlArray{T}, shape::MPS.MPSShape) where {T} =
    MPSGraphTensorData(matrix.data[], shape, T)

# Following Laya, encode the cached MPSGraph into Metal's current batch instead
# of committing a separate command buffer (and flushing kernels) per product.
# All arrays and Objective-C feed objects stay rooted until GPU completion.
@autoreleasepool function batched_matmul!(c, a, b, transpose_a, transpose_b)
    key = MatmulGraphKey(a, b, c, true, false, transpose_a, transpose_b)
    cached = @lock PRODUCT_GRAPH_LOCK get!(PRODUCT_GRAPH_CACHE, key) do
        ProductGraph(key)
    end
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        # The MtlArray constructor rebuilds collect/NSNumber/NSArray shape
        # metadata on every call. Shapes are immutable and already graph-keyed.
        cached.place_a => graph_tensor_data(a, cached.shape_a),
        cached.place_b => graph_tensor_data(b, cached.shape_b),
    )
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.result => graph_tensor_data(c, cached.shape_c),
    )
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
include("metal_rope.jl")
include("metal_attention.jl")

end
