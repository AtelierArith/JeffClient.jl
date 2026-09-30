# Adapted from ../Laya.jl/ext/LayaMetalExt.jl. Buffers are returned by their last
# DataRef owner, including Metal's queued-operation roots, rather than freed
# while commands can still access them. Reuse is restricted to the same queue.
const BUFFER_PAGE = 16384
const BUFFER_POOL = Dict{Tuple{UInt,DataType,Int},Vector{Metal.MTLBuffer}}()
const BUFFER_POOL_LOCK = ReentrantLock()
const BUFFER_POOL_BYTES = Dict{UInt,Int}()
const BUFFER_POOL_LIMITS = Dict{UInt,Int}()
const BUFFER_MISSES = Ref(0)
const BUFFER_REUSES = Ref(0)

struct ReturnBuffer
    key::Tuple{UInt,DataType,Int}
end

function (owner::ReturnBuffer)(buffer::Metal.MTLBuffer)
    @lock BUFFER_POOL_LOCK begin
        push!(get!(Vector{Metal.MTLBuffer}, BUFFER_POOL, owner.key), buffer)
        queue = owner.key[1]
        BUFFER_POOL_BYTES[queue] = get(BUFFER_POOL_BYTES, queue, 0) + owner.key[3]
    end
    return nothing
end

function clear_buffer_pool!()
    queue = Metal.global_queue(Metal.device())
    Metal.synchronize(queue)
    id = objectid(queue)
    buffers = @lock BUFFER_POOL_LOCK begin
        collected = Metal.MTLBuffer[]
        for key in collect(keys(BUFFER_POOL))
            key[1] == id || continue
            append!(collected, pop!(BUFFER_POOL, key))
        end
        BUFFER_POOL_BYTES[id] = 0
        collected
    end
    foreach(Metal.free, buffers)
    return nothing
end

# Call only after completing this queue's GPU work. A miss-only pressure check
# can leave an oversized pool intact indefinitely when all sizes are reused.
function trim_completed_buffer_pool!()
    queue = objectid(Metal.global_queue(Metal.device()))
    buffers = @lock BUFFER_POOL_LOCK begin
        bytes = get(BUFFER_POOL_BYTES, queue, 0)
        limit = get(BUFFER_POOL_LIMITS, queue, typemax(Int))
        bytes <= limit && return nothing
        retired = Metal.MTLBuffer[]
        keys_by_size = sort(
            [key for key in keys(BUFFER_POOL) if key[1] == queue];
            by = last,
            rev = true,
        )
        for key in keys_by_size
            free = BUFFER_POOL[key]
            while bytes > limit && !isempty(free)
                push!(retired, pop!(free))
                bytes -= key[3]
            end
            bytes <= limit && break
        end
        BUFFER_POOL_BYTES[queue] = bytes
        retired
    end
    foreach(Metal.free, buffers)
    return nothing
end

function pooled_array(::Type{T}, dims::Dims{N}) where {T,N}
    queue = objectid(Metal.global_queue(Metal.device()))
    bytes = cld(max(prod(dims) * sizeof(T), 1), BUFFER_PAGE) * BUFFER_PAGE
    key = (queue, T, bytes)
    buffer, excessive = @lock BUFFER_POOL_LOCK begin
        free = get(BUFFER_POOL, key, nothing)
        limit = get!(BUFFER_POOL_LIMITS, queue) do
            Int(Metal.device().recommendedMaxWorkingSetSize) ÷ 4
        end
        if free === nothing || isempty(free)
            BUFFER_MISSES[] += 1
            nothing, get(BUFFER_POOL_BYTES, queue, 0) > limit
        else
            BUFFER_REUSES[] += 1
            BUFFER_POOL_BYTES[queue] -= bytes
            pop!(free), false
        end
    end
    excessive && clear_buffer_pool!()
    buffer === nothing &&
        (buffer = Metal.alloc(Metal.device(), bytes; storage = Metal.PrivateStorage))
    reference = Metal.GPUArrays.DataRef(ReturnBuffer(key), buffer)
    array = Metal.MtlArray{T,N,Metal.PrivateStorage}(reference, dims; maxsize = bytes)
    Metal.GPUArrays.unsafe_free!(reference) # The array now owns the reference.
    return array
end

function metal_pool_stats()
    queue = objectid(Metal.global_queue(Metal.device()))
    @lock BUFFER_POOL_LOCK return (
        misses = BUFFER_MISSES[],
        reuses = BUFFER_REUSES[],
        free_bytes = get(BUFFER_POOL_BYTES, queue, 0),
        limit_bytes = get(BUFFER_POOL_LIMITS, queue, 0),
        free_buckets = sort(
            [
                (
                    buffer_bytes = key[3],
                    buffers = length(buffers),
                    total_bytes = key[3] * length(buffers),
                ) for (key, buffers) in BUFFER_POOL if key[1] == queue && !isempty(buffers)
            ];
            by = bucket -> bucket.total_bytes,
            rev = true,
        ),
        upload_misses = UPLOAD_MISSES[],
        upload_reuses = UPLOAD_REUSES[],
        upload_pending_bytes = get(UPLOAD_PENDING_BYTES, queue, 0),
    )
end

# Shared uploads avoid Metal's staging copy and its queue-wide synchronization.
# Unlike device-only buffers, the host must not rewrite these while the GPU may
# read them. A last-owner finalizer puts them in a pending list; only a completed
# download or explicit synchronization moves this queue's buffers to the free list.
const UPLOAD_POOL = Dict{Tuple{UInt,DataType,Int},Vector{Metal.MTLBuffer}}()
const UPLOAD_PENDING = Tuple{Tuple{UInt,DataType,Int},Metal.MTLBuffer}[]
const UPLOAD_PENDING_BYTES = Dict{UInt,Int}()
const UPLOAD_LIMIT = 64 << 20
const UPLOAD_MISSES = Ref(0)
const UPLOAD_REUSES = Ref(0)

struct ReturnUpload
    key::Tuple{UInt,DataType,Int}
end

function (owner::ReturnUpload)(buffer::Metal.MTLBuffer)
    @lock BUFFER_POOL_LOCK begin
        push!(UPLOAD_PENDING, (owner.key, buffer))
        queue = owner.key[1]
        UPLOAD_PENDING_BYTES[queue] = get(UPLOAD_PENDING_BYTES, queue, 0) + owner.key[3]
    end
    return nothing
end

function recycle_uploads!()
    queue = objectid(Metal.global_queue(Metal.device()))
    @lock BUFFER_POOL_LOCK begin
        for (key, buffer) in UPLOAD_PENDING
            key[1] == queue || continue
            push!(get!(Vector{Metal.MTLBuffer}, UPLOAD_POOL, key), buffer)
        end
        filter!(entry -> first(entry)[1] != queue, UPLOAD_PENDING)
        UPLOAD_PENDING_BYTES[queue] = 0
        # The queue is idle here; bound retained shared buffers across changing
        # input lengths, without freeing another task's in-flight resources.
        retained = sum(
            key[3] * length(buffers) for (key, buffers) in UPLOAD_POOL if key[1] == queue;
            init = 0,
        )
        if retained > UPLOAD_LIMIT
            for key in collect(keys(UPLOAD_POOL))
                key[1] == queue || continue
                foreach(Metal.free, pop!(UPLOAD_POOL, key))
            end
        end
    end
    return nothing
end

function JeffClient.on_native_device(
    ::Metal.MtlArray,
    input::AbstractArray{T,N},
) where {T,N}
    host = convert(Array{T,N}, input)
    queue = objectid(Metal.global_queue(Metal.device()))
    bytes = cld(max(sizeof(host), 1), BUFFER_PAGE) * BUFFER_PAGE
    key = (queue, T, bytes)
    function take_buffer()
        @lock BUFFER_POOL_LOCK begin
            free = get(UPLOAD_POOL, key, nothing)
            if free === nothing || isempty(free)
                nothing
            else
                UPLOAD_REUSES[] += 1
                pop!(free)
            end
        end
    end
    buffer = take_buffer()
    pending = @lock BUFFER_POOL_LOCK get(UPLOAD_PENDING_BYTES, queue, 0)
    if buffer === nothing && pending > UPLOAD_LIMIT
        Metal.synchronize()
        recycle_uploads!()
        buffer = take_buffer()
    end
    if buffer === nothing
        @lock BUFFER_POOL_LOCK UPLOAD_MISSES[] += 1
        buffer = Metal.alloc(Metal.device(), bytes; storage = Metal.SharedStorage)
    end
    GC.@preserve host unsafe_copyto!(
        convert(Ptr{T}, Metal.MTL.contents(buffer)),
        pointer(host),
        length(host),
    )
    reference = Metal.GPUArrays.DataRef(ReturnUpload(key), buffer)
    array = Metal.MtlArray{T,N,Metal.SharedStorage}(reference, size(host); maxsize = bytes)
    Metal.GPUArrays.unsafe_free!(reference)
    return array
end

function JeffClient.native_host(input::Metal.MtlArray)
    host = Array(input) # This waits for the current queue's GPU work.
    recycle_uploads!()
    trim_completed_buffer_pool!()
    return host
end
