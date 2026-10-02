# A model's embedding identifies its private scratch. Weak owners keep the
# cache from retaining model weights. Same-backend forwards are serialized;
# scratch is never handed to another forward before its stream has completed.
mutable struct ForwardWorkspace
    lock::ReentrantLock
    stream::CUDA.CuStream
    slots::Vector{Any}
    layer_slots::Dict{Symbol,Vector{Any}}
    active_slots::Vector{Any}
    cursor::Int
    one::CUDA.CuVector{Float32,CUDA.DeviceMemory}
    zero::CUDA.CuVector{Float32,CUDA.DeviceMemory}
end
const WORKSPACES = Dict{UInt,Tuple{WeakRef,ForwardWorkspace}}()
const WORKSPACE_LOCK = ReentrantLock()
const WORKSPACE_KEY = :jeff_native_cuda_workspace

function workspace(reference)
    @lock WORKSPACE_LOCK begin
        for key in collect(keys(WORKSPACES))
            WORKSPACES[key][1].value === nothing && delete!(WORKSPACES, key)
        end
        key = objectid(reference)
        entry = get(WORKSPACES, key, nothing)
        if entry === nothing || entry[1].value !== reference
            slots = Any[]
            owner = ForwardWorkspace(
                ReentrantLock(),
                CUDA.stream(),
                slots,
                Dict{Symbol,Vector{Any}}(),
                slots,
                0,
                CUDA.CuArray{Float32,1,CUDA.DeviceMemory}(Float32[1]),
                CUDA.CuArray{Float32,1,CUDA.DeviceMemory}(Float32[0]),
            )
            WORKSPACES[key] = (WeakRef(reference), owner)
            return owner
        end
        return entry[2]
    end
end

function JeffClient.native_forward_scope(f::F, reference::CUDA.CuArray) where {F}
    if CUDA.device() != CUDA.device(reference)
        return CUDA.device!(CUDA.device(reference)) do
            JeffClient.native_forward_scope(f, reference)
        end
    end
    owner = workspace(reference)
    lock(owner.lock) do
        CUDA.synchronize(owner.stream)
        owner.stream = CUDA.stream()
        owner.cursor = 0
        owner.active_slots = owner.slots
        task_local_storage(WORKSPACE_KEY, owner) do
            try
                return f()
            finally
                # Covers failed forwards as well as ordinary CPU readback.
                CUDA.synchronize(owner.stream)
            end
        end
    end
end

function scratch(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    owner = get(task_local_storage(), WORKSPACE_KEY, nothing)
    owner === nothing && return CUDA.CuArray{T,N,CUDA.DeviceMemory}(undef, dims)
    owner = owner::ForwardWorkspace
    owner.cursor += 1
    slot = owner.cursor
    slots = owner.active_slots
    if slot > length(slots)
        push!(slots, CUDA.CuArray{T,N,CUDA.DeviceMemory}(undef, dims))
    else
        array = slots[slot]
        if !(array isa CUDA.CuArray{T,N,CUDA.DeviceMemory}) || size(array) != dims
            slots[slot] = CUDA.CuArray{T,N,CUDA.DeviceMemory}(undef, dims)
        end
    end
    return slots[slot]::CUDA.CuArray{T,N,CUDA.DeviceMemory}
end

const DeviceMatrix =
    Union{CUDA.StridedCuMatrix{Float32},Transpose{Float32,<:CUDA.StridedCuMatrix{Float32}}}
function JeffClient.native_matmul(a::DeviceMatrix, b::DeviceMatrix)
    output = scratch(Float32, (size(a, 1), size(b, 2)))
    owner = get(task_local_storage(), WORKSPACE_KEY, nothing)
    owner === nothing && return mul!(output, a, b)
    owner = owner::ForwardWorkspace
    left = a isa Transpose ? parent(a) : a
    right = b isa Transpose ? parent(b) : b
    ta, tb = a isa Transpose ? 'T' : 'N', b isa Transpose ? 'T' : 'N'
    m, n, k = size(output, 1), size(output, 2), size(a, 2)
    size(b, 1) == k || throw(DimensionMismatch("Matrix inner dimensions must match."))
    blas = CUDA.CUBLAS
    compute = blas.gemmExComputeType(Float32, Float32, Float32, m, k, n)
    # CUDA.jl 6.4.1's gemmEx! constructs two device CuRefs on every call.
    # Match its checked low-level call with model-owned, immutable coefficients.
    # Pointer mode remains DEVICE, as configured by CUDA.jl's task-local handle.
    blas.cublasGemmEx(
        blas.handle(),
        ta,
        tb,
        m,
        n,
        k,
        owner.one,
        left,
        Float32,
        max(1, stride(left, 2)),
        right,
        Float32,
        max(1, stride(right, 2)),
        owner.zero,
        output,
        Float32,
        max(1, stride(output, 2)),
        compute,
        blas.CUBLAS_GEMM_DEFAULT,
    )
    return output
end

function batched_matmul!(output, a, b, ta, tb)
    owner = get(task_local_storage(), WORKSPACE_KEY, nothing)
    if owner === nothing
        return CUDA.CUBLAS.gemm_strided_batched!(ta, tb, 1.0f0, a, b, 0.0f0, output)
    end
    owner = owner::ForwardWorkspace
    return CUDA.CUBLAS.gemm_strided_batched!(ta, tb, owner.one, a, b, owner.zero, output)
end

function rms_kernel!(output, input, weight, width, columns, eps, centered)
    lane = Int32(CUDA.threadIdx().x)
    column = Int32(CUDA.blockIdx().x)
    total = 0.0f0
    for row = lane:Int32(32):width
        value = input[row+(column-Int32(1))*width]
        total += value*value
    end
    scale = inv(sqrt(warp_sum(total)/Float32(width)+eps))
    for row = lane:Int32(32):width
        offset = row+(column-Int32(1))*width
        gamma = centered ? 1.0f0+weight[row] : weight[row]
        output[offset] = input[offset]*scale*gamma
    end
    return
end

function JeffClient.native_rms(input::CUDA.CuArray{Float32}, weight, eps; centered = true)
    output = scratch(Float32, size(input))
    width = size(input, 1)
    columns = length(input)÷width
    CUDA.@cuda threads=32 blocks=columns rms_kernel!(
        output,
        input,
        weight,
        Int32(width),
        Int32(columns),
        eps,
        centered,
    )
    return output
end

function JeffClient.native_prepare_mask(reference::CUDA.CuArray, mask)
    output = scratch(Int32, (length(mask),))
    copyto!(output, Int32.(mask))
    return output
end

function JeffClient.native_residual_rms(x::CUDA.CuMatrix{Float32}, mixed, weight, eps)
    residual = scratch(Float32, size(x))
    residual .= x .+ mixed
    return residual, JeffClient.native_rms(residual, weight, eps)
end

function JeffClient.native_hidden_forward(
    hidden::CUDA.CuMatrix{Float32},
    layers,
    mask,
    final_norm,
    cfg,
)
    owner = get(task_local_storage(), WORKSPACE_KEY, nothing)
    if owner === nothing
        for layer in layers
            hidden = JeffClient.native_layer(layer, hidden, mask, cfg)
        end
    else
        owner = owner::ForwardWorkspace
        buffers = (scratch(Float32, size(hidden)), scratch(Float32, size(hidden)))
        root_slots, root_cursor = owner.active_slots, owner.cursor
        for (index, layer) in enumerate(layers)
            owner.active_slots = get!(()->Any[], owner.layer_slots, layer.attention.kind)
            owner.cursor = 0
            destination = buffers[isodd(index) ? 1 : 2]
            try
                result = JeffClient.native_layer(layer, hidden, mask, cfg)
                # Preserve the next layer's input outside the reused layer
                # scratch. The copy and subsequent writes share one stream.
                copyto!(destination, result)
            finally
                owner.active_slots, owner.cursor = root_slots, root_cursor
            end
            hidden = destination
        end
    end
    return JeffClient.native_rms(
        view(hidden, :, size(hidden, 2):size(hidden, 2)),
        final_norm,
        cfg.eps,
    )
end

function gather_kernel!(output, embedding, ids, width, elements)
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(
            CUDA.threadIdx().x,
        )
    if index <= elements
        row = rem(index-Int32(1), width)+Int32(1)
        token = (index-Int32(1))÷width+Int32(1)
        @inbounds output[index] = embedding[row, ids[token]+Int32(1)]
    end
    return
end

function JeffClient.native_gather(embedding::CUDA.CuMatrix{Float32}, ids::Vector{<:Integer})
    all(id -> 0 <= id < size(embedding, 2), ids) ||
        throw(ArgumentError("Token ID is outside the vocabulary."))
    device_ids = scratch(Int32, (length(ids),))
    copyto!(device_ids, Int32.(ids))
    output = scratch(Float32, (size(embedding, 1), length(ids)))
    CUDA.@cuda threads=256 blocks=cld(length(output), 256) gather_kernel!(
        output,
        embedding,
        device_ids,
        Int32(size(embedding, 1)),
        Int32(length(output)),
    )
    return output
end
