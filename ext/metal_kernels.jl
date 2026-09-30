# Metal caches compilation but still constructs a pipeline Ref on each lookup.
# Cache only singleton kernels with default compiler options, and discard our
# handles on any Julia method update. Launch through Metal's ordinary callable
# kernel so argument encoding, queue roots, and synchronization stay intact.
const KERNEL_HANDLE_LOCK = ReentrantLock()
const KERNEL_HANDLES = Dict{Tuple{UInt,DataType,DataType},Any}()
const KERNEL_HANDLE_WORLD = Ref{UInt}(0)

function cached_mtlfunction(f::F, ::Type{TT}) where {F,TT}
    Base.issingletontype(F) || return Metal.mtlfunction(f, TT)
    device = Metal.device()
    world = Base.get_world_counter()
    key = (objectid(device), F, TT)
    @lock KERNEL_HANDLE_LOCK begin
        if KERNEL_HANDLE_WORLD[] != world
            empty!(KERNEL_HANDLES)
            KERNEL_HANDLE_WORLD[] = world
        end
        kernel = get(KERNEL_HANDLES, key, nothing)
        if kernel === nothing
            kernel = Metal.mtlfunction(f, TT)
            KERNEL_HANDLES[key] = kernel
        end
        return kernel::Metal.HostKernel{F,TT}
    end
end

function launch_cached_kernel!(f::F, args::Vararg{Any,N}; threads, groups) where {F,N}
    GC.@preserve f args begin
        kernel_f = Metal.mtlconvert(f)
        kernel_args = map(Metal.mtlconvert, args)
        kernel_type = Tuple{map(Core.Typeof, kernel_args)...}
        kernel = cached_mtlfunction(kernel_f, kernel_type)
        kernel(args...; threads, groups)
    end
    return nothing
end
