module JeffClientMetalExt

import JeffClient
import Metal
using LinearAlgebra
using Metal: MPS
using Metal.ObjectiveC: @objc, id
using Metal.MPSGraphs:
    MatmulGraphKey,
    MPSGraph,
    MPSGraphTensor,
    MPSGraphTensorData,
    placeholderTensor,
    transposeTensor,
    matrixMultiplicationWithPrimaryTensor,
    default_exec_desc
using Metal.ObjectiveC.Foundation:
    @autoreleasepool, NSArray, NSDictionary, NSUInteger, nil, retain, release

include("metal_buffers.jl")

function JeffClient.native_array(::Val{:metal}, x)
    Metal.functional() || throw(ArgumentError("Metal is not available on this machine."))
    return Metal.MtlArray(x)
end

function embedding_gather_kernel!(output, embedding, ids, width, elements)
    index = Int(Metal.thread_position_in_grid_1d())
    if index <= elements
        token = (index - 1) ÷ width + 1
        channel = (index - 1) % width + 1
        @inbounds output[index] = embedding[channel, ids[token]]
    end
    return
end

function JeffClient.native_gather(
    embedding::Metal.MtlMatrix{Float32},
    ids::Vector{<:Integer},
)
    all(id -> 0 <= id < size(embedding, 2), ids) ||
        throw(ArgumentError("Token ID is outside the vocabulary."))
    device_ids = JeffClient.on_native_device(embedding, Int32.(ids .+ 1))
    output = pooled_array(Float32, (size(embedding, 1), length(ids)))
    elements = length(output)
    if elements > 0
        Metal.@metal threads=256 groups=cld(elements, 256) embedding_gather_kernel!(
            output,
            embedding,
            device_ids,
            size(embedding, 1),
            elements,
        )
    end
    return output
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
    feed_keys::NSArray
    result_keys::NSArray
    feed_key_ids::Base.RefValue{NTuple{2,id{MPSGraphTensor}}}
    result_key_ids::Base.RefValue{NTuple{1,id{MPSGraphTensor}}}
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
    feed_keys = NSArray([place_a, place_b])
    result_keys = NSArray([result])
    cached = ProductGraph(
        graph,
        place_a,
        place_b,
        result,
        shape_a,
        shape_b,
        shape_c,
        feed_keys,
        result_keys,
        Ref((pointer(place_a), pointer(place_b))),
        Ref((pointer(result),)),
    )
    # NSArray is an unmanaged autoreleased wrapper in ObjectiveC.jl. Julia's
    # cache alone cannot keep its underlying object alive beyond this pool.
    retain(shape_a)
    retain(shape_b)
    retain(shape_c)
    retain(feed_keys)
    retain(result_keys)
    finalizer(cached) do owner
        release(owner.shape_a)
        release(owner.shape_b)
        release(owner.shape_c)
        release(owner.feed_keys)
        release(owner.result_keys)
    end
    return cached
end

const PRODUCT_GRAPH_CACHE = Dict{MatmulGraphKey,ProductGraph}()
const PRODUCT_GRAPH_LOCK = ReentrantLock()

# Only immutable model-weight bindings use this cache. Weak owners avoid
# retaining the Julia model; their finalizers remove the native tensor-data.
const WEIGHT_TENSOR_CACHE = Dict{UInt,Tuple{WeakRef,MPSGraphTensorData}}()
const WEIGHT_TENSOR_LOCK = ReentrantLock()

function weight_tensor_data(matrix, shape)
    key = objectid(matrix)
    @lock WEIGHT_TENSOR_LOCK begin
        entry = get(WEIGHT_TENSOR_CACHE, key, nothing)
        entry !== nothing && entry[1].value === matrix && return entry[2]
        data = graph_tensor_data(matrix, shape)
        WEIGHT_TENSOR_CACHE[key] = (WeakRef(matrix), data)
        finalizer(matrix) do owner
            @lock WEIGHT_TENSOR_LOCK begin
                current = get(WEIGHT_TENSOR_CACHE, key, nothing)
                if current !== nothing &&
                   (current[1].value === owner || current[1].value === nothing)
                    pop!(WEIGHT_TENSOR_CACHE, key, nothing)
                end
            end
        end
        return data
    end
end

function JeffClient.native_linear(
    weight::Metal.MtlMatrix{Float32},
    x::Metal.MtlMatrix{Float32},
)
    size(weight, 1) == size(x, 1) ||
        throw(DimensionMismatch("Linear input width must match."))
    output = pooled_array(Float32, (size(weight, 2), size(x, 2)))
    return batched_matmul!(output, weight, x, 'T', 'N'; immutable_a = true)
end

function graph_tensor_data(matrix::Metal.MtlArray{T}, shape::MPS.MPSShape) where {T}
    workspace = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    if workspace isa ForwardWorkspace && workspace.active
        key = objectid(matrix)
        cached = get(workspace.tensor_data, key, nothing)
        cached !== nothing && return cached[1]
        # Only cache owned slots, whose buffer and physical dimensions are fixed.
        # Temporary reshape wrappers and standalone arrays keep the usual path.
        if workspace.cursor > 0 && workspace.slots[workspace.cursor] === matrix
            data = MPSGraphTensorData(matrix.data[], shape, T)
            workspace.tensor_data[key] = MPSGraphTensorData[data]
            return data
        end
    end
    return MPSGraphTensorData(matrix.data[], shape, T)
end

function result_tensor_values(matrix, shape)
    data = graph_tensor_data(matrix, shape)
    workspace = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    if workspace isa ForwardWorkspace && workspace.active
        cached = get(workspace.tensor_data, objectid(matrix), nothing)
        cached !== nothing && return cached
    end
    return MPSGraphTensorData[data]
end

function tensor_dictionary(
    keys::Base.RefValue{NTuple{N,id{MPSGraphTensor}}},
    values::Vector{MPSGraphTensorData},
    ::Val{N},
) where {N}
    length(values) == N ||
        throw(DimensionMismatch("Tensor dictionary value count mismatch."))
    identifiers = ntuple(i -> pointer(values[i]), Val(N))
    storage = Ref(identifiers)
    dictionary = GC.@preserve values storage keys begin
        address = Ptr{id{MPSGraphTensorData}}(
            Base.unsafe_convert(Ptr{typeof(identifiers)}, storage),
        )
        key_address = Ptr{id{MPSGraphTensor}}(
            Base.unsafe_convert(Ptr{NTuple{N,id{MPSGraphTensor}}}, keys),
        )
        @objc [
            NSDictionary dictionaryWithObjects:(address::Ptr{id{MPSGraphTensorData}})
            forKeys:(key_address::Ptr{id{MPSGraphTensor}})
            count:(N::NSUInteger)
        ]::id{NSDictionary}
    end
    return NSDictionary(dictionary)
end

# Following Laya, encode the cached MPSGraph into Metal's current batch instead
# of committing a separate command buffer (and flushing kernels) per product.
# All arrays and Objective-C feed objects stay rooted until GPU completion.
@autoreleasepool function batched_matmul!(
    c,
    a,
    b,
    transpose_a,
    transpose_b;
    immutable_a = false,
)
    key = MatmulGraphKey(a, b, c, true, false, transpose_a, transpose_b)
    cached = @lock PRODUCT_GRAPH_LOCK get!(PRODUCT_GRAPH_CACHE, key) do
        ProductGraph(key)
    end
    # Fixed keys and shapes belong to the graph. Only the tensor-data values
    # change per product; avoid Julia Dict and its keys/values conversion copies.
    feed_values = MPSGraphTensorData[
        immutable_a ? weight_tensor_data(a, cached.shape_a) :
        graph_tensor_data(a, cached.shape_a),
        graph_tensor_data(b, cached.shape_b),
    ]
    result_values = result_tensor_values(c, cached.shape_c)
    feeds = tensor_dictionary(cached.feed_key_ids, feed_values, Val(2))
    results = tensor_dictionary(cached.result_key_ids, result_values, Val(1))
    queue = Metal.global_queue(Metal.device())
    Metal.end_encoder!(queue)
    command = MPS.MPSCommandBuffer(Metal.ensure_cmdbuf!(queue))
    MPS.encode!(command, cached.graph, feeds, results, nil, default_exec_desc())
    # encode! consumes the autoreleased dictionaries inside this pool. Their
    # managed tensor-data values remain rooted for the queued GPU operations,
    # exactly as when those values were held in Julia dictionaries.
    Metal.record_operation!(queue, a, b, c, feed_values, result_values, command)
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

struct PreparedMetalMask{H,D}
    host::H
    device::D
end

function JeffClient.native_prepare_mask(reference::Metal.MtlArray, mask)
    PreparedMetalMask(mask, JeffClient.on_native_device(reference, Float32.(mask)))
end

metal_device_mask(reference, mask) = JeffClient.on_native_device(reference, Float32.(mask))
metal_device_mask(reference, mask::PreparedMetalMask) = mask.device
metal_host_mask(mask) = mask
metal_host_mask(mask::PreparedMetalMask) = mask.host

include("metal_delta.jl")
include("metal_normalization.jl")
include("metal_softmax.jl")
include("metal_rope.jl")
include("metal_attention.jl")

end
