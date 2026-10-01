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

function fresh_pooled_array(::Type{T}, dims::Dims{N}) where {T,N}
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

# Every allocation position owns a distinct array, even when dimensions match.
# Keep only the last execution's slots; reuse starts after its CPU readback.
mutable struct ForwardWorkspace
    queue::UInt
    command_queue::Metal.BatchedCommandQueue
    slots::Vector{Any}
    slot_indices::Dict{UInt,Int}
    tensor_data::Dict{UInt,Vector{MPSGraphTensorData}}
    feed_values::Dict{UInt,Vector{MPSGraphTensorData}}
    cursor::Int
    active::Bool
    bytes::Int
end

const FORWARD_WORKSPACE_KEY = :JeffClientMetalForwardWorkspace
const SHAPE_WORKSPACE_KEY = :JeffClientMetalShapeWorkspaces

mutable struct ShapeWorkspaces
    queue::UInt
    entries::Dict{Int,ForwardWorkspace}
    order::Vector{Int}
    byte_limit::Int
end

new_forward_workspace(queue) = ForwardWorkspace(
    queue,
    Metal.global_queue(Metal.device()),
    Any[],
    Dict{UInt,Int}(),
    Dict{UInt,Vector{MPSGraphTensorData}}(),
    Dict{UInt,Vector{MPSGraphTensorData}}(),
    0,
    false,
    0,
)

function release_workspace!(workspace::ForwardWorkspace)
    workspace.active && throw(ArgumentError("Cannot release an active forward workspace."))
    empty!(workspace.tensor_data)
    empty!(workspace.feed_values)
    empty!(workspace.slots)
    empty!(workspace.slot_indices)
    workspace.bytes = 0
    return nothing
end

function clear_forward_workspace!()
    workspace = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    bank = get(task_local_storage(), SHAPE_WORKSPACE_KEY, nothing)
    workspace isa ForwardWorkspace &&
        workspace.active &&
        throw(ArgumentError("Cannot clear an active forward workspace."))
    if bank isa ShapeWorkspaces
        any(entry.active for entry in values(bank.entries)) &&
            throw(ArgumentError("Cannot clear active shape workspaces."))
    end
    !(workspace isa ForwardWorkspace) && !(bank isa ShapeWorkspaces) && return nothing
    Metal.synchronize()
    workspace isa ForwardWorkspace && release_workspace!(workspace)
    if bank isa ShapeWorkspaces
        foreach(release_workspace!, values(bank.entries))
        empty!(bank.entries)
        empty!(bank.order)
        delete!(task_local_storage(), SHAPE_WORKSPACE_KEY)
    end
    delete!(task_local_storage(), FORWARD_WORKSPACE_KEY)
    return nothing
end

function pooled_array(::Type{T}, dims::Dims{N}) where {T,N}
    workspace = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    if workspace isa ForwardWorkspace && workspace.active
        workspace.cursor += 1
        slot = workspace.cursor
        if slot <= length(workspace.slots)
            cached = workspace.slots[slot]
            if cached isa Metal.MtlArray{T,N,Metal.PrivateStorage} && size(cached) == dims
                return cached
            end
        end
        array = fresh_pooled_array(T, dims)
        if slot <= length(workspace.slots)
            workspace.bytes -= (workspace.slots[slot]::Metal.MtlArray).maxsize
            delete!(workspace.slot_indices, objectid(workspace.slots[slot]))
            pop!(workspace.tensor_data, objectid(workspace.slots[slot]), nothing)
            pop!(workspace.feed_values, objectid(workspace.slots[slot]), nothing)
            workspace.slots[slot] = array
        else
            push!(workspace.slots, array)
        end
        workspace.slot_indices[objectid(array)] = slot
        workspace.bytes += array.maxsize
        return array
    end
    return fresh_pooled_array(T, dims)
end

function JeffClient.native_forward_scope(f, reference::Metal.MtlArray)
    # Opt in while comparing the prototype's retained memory and allocation cost.
    get(ENV, "JEFF_METAL_WORKSPACE", "0") == "1" || return f()
    queue = objectid(Metal.global_queue(Metal.device()))
    workspace = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    if !(workspace isa ForwardWorkspace) || workspace.queue != queue
        workspace = new_forward_workspace(queue)
        task_local_storage(FORWARD_WORKSPACE_KEY, workspace)
    end
    workspace.active && return f() # Nested work must keep advancing the outer slots.
    workspace.cursor = 0
    workspace.active = true
    try
        return f() # The row ends with native_host, which waits for GPU completion.
    catch
        Metal.synchronize()
        rethrow()
    finally
        workspace.active = false
        for slot = (workspace.cursor+1):length(workspace.slots)
            workspace.bytes -= (workspace.slots[slot]::Metal.MtlArray).maxsize
            delete!(workspace.slot_indices, objectid(workspace.slots[slot]))
            pop!(workspace.tensor_data, objectid(workspace.slots[slot]), nothing)
            pop!(workspace.feed_values, objectid(workspace.slots[slot]), nothing)
        end
        resize!(workspace.slots, workspace.cursor)
        # GPU completion precedes this cleanup. Keep reusable containers but
        # replace input/weight bindings with an already-owned output binding.
        for (key, values) in workspace.feed_values
            fill!(values, workspace.tensor_data[key][1])
        end
    end
end

function evict_shape_workspace!(bank::ShapeWorkspaces)
    key = popfirst!(bank.order)
    release_workspace!(pop!(bank.entries, key))
    return nothing
end

function JeffClient.native_forward_scope(
    f::F,
    reference::Metal.MtlArray,
    sequence_length::Int,
) where {F}
    current = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    current isa ForwardWorkspace &&
        current.active &&
        return JeffClient.native_forward_scope(f, reference)
    enabled =
        get(ENV, "JEFF_METAL_WORKSPACE", "0") == "1" &&
        get(ENV, "JEFF_METAL_SHAPE_WORKSPACES", "0") == "1"
    bank = get(task_local_storage(), SHAPE_WORKSPACE_KEY, nothing)
    if !enabled
        bank isa ShapeWorkspaces && clear_forward_workspace!()
        return JeffClient.native_forward_scope(f, reference)
    end
    queue = objectid(Metal.global_queue(Metal.device()))
    if !(bank isa ShapeWorkspaces) || bank.queue != queue
        clear_forward_workspace!()
        bank = ShapeWorkspaces(
            queue,
            Dict{Int,ForwardWorkspace}(),
            Int[],
            Int(Metal.device().recommendedMaxWorkingSetSize) ÷ 4,
        )
        task_local_storage(SHAPE_WORKSPACE_KEY, bank)
    end
    index = findfirst(==(sequence_length), bank.order)
    index !== nothing && deleteat!(bank.order, index)
    push!(bank.order, sequence_length)
    workspace = get!(bank.entries, sequence_length) do
        new_forward_workspace(queue)
    end
    while length(bank.order) > 2
        evict_shape_workspace!(bank)
    end
    task_local_storage(FORWARD_WORKSPACE_KEY, workspace)
    try
        return JeffClient.native_forward_scope(f, reference)
    finally
        # A complete readback or the inner exception handler precedes eviction.
        while length(bank.order) > 1 &&
            sum(entry.bytes for entry in values(bank.entries)) > bank.byte_limit
            evict_shape_workspace!(bank)
        end
    end
end

function metal_pool_stats()
    queue = objectid(Metal.global_queue(Metal.device()))
    workspace = get(task_local_storage(), FORWARD_WORKSPACE_KEY, nothing)
    bank = get(task_local_storage(), SHAPE_WORKSPACE_KEY, nothing)
    workspaces =
        bank isa ShapeWorkspaces ? collect(values(bank.entries)) :
        workspace isa ForwardWorkspace ? [workspace] : ForwardWorkspace[]
    workspace_arrays = sum(length(entry.slots) for entry in workspaces; init = 0)
    workspace_bytes = sum(entry.bytes for entry in workspaces; init = 0)
    @lock BUFFER_POOL_LOCK return (
        misses = BUFFER_MISSES[],
        reuses = BUFFER_REUSES[],
        free_bytes = get(BUFFER_POOL_BYTES, queue, 0),
        workspace_arrays = workspace_arrays,
        workspace_feed_vectors = sum(
            length(entry.feed_values) for entry in workspaces;
            init = 0,
        ),
        workspace_bytes = workspace_bytes,
        workspace_tensor_data = sum(
            length(entry.tensor_data) for entry in workspaces;
            init = 0,
        ),
        workspace_cached_sequence_lengths = bank isa ShapeWorkspaces ?
                                            sort(collect(keys(bank.entries))) : Int[],
        workspace_current_arrays = workspace isa ForwardWorkspace ?
                                   length(workspace.slots) : 0,
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
