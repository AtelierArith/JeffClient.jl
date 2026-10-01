# Automatic, immutable CPU policy. Overrides are internal diagnostic scopes,
# not environment-variable configuration or mutable backend caches.
function cpu_defaults(apple::Bool)
    portable = !apple
    return (
        accelerate = apple,
        vector_math = apple,
        portable_vector_math = portable,
        vector_math_blocks = true,
        parallel_projections = portable,
        projection_thread_scope = true,
        parallel_full_heads = portable,
        recurrent_delta = portable,
        delta_norm_loop = portable,
        delta_workspace = portable,
        delta_projection_workspace = portable,
        final_query = portable,
        mlp_residual_fusion = portable,
        parallel_heads = true,
        mlp_workspace = true,
        trim_padding = true,
        inplace_delta_rms = false,
        simd = false,
        octavian_delta = false,
        delta_chunk_size = 64,
        delta_workers = Threads.nthreads(:default),
    )
end
const CPU_DEFAULTS = Ref(cpu_defaults(false))
cpu_settings() =
    get(task_local_storage(), :jeff_cpu_settings, CPU_DEFAULTS[])::typeof(CPU_DEFAULTS[])
@inline cpu_setting(name::Symbol) = getproperty(cpu_settings(), name)

with_cpu_settings(f::F, settings::NamedTuple) where {F} =
    task_local_storage(f, :jeff_cpu_settings, settings)
function with_cpu_settings(f::F, overrides::Vararg{Pair,N}) where {F,N}
    names = map(first, overrides)
    values = map(overrides) do pair
        name, value = pair
        expected = getproperty(CPU_DEFAULTS[], name)
        value isa AbstractString ?
        (expected isa Bool ? value == "1" : parse(Int, value)) : value
    end
    return with_cpu_settings(f, merge(cpu_settings(), NamedTuple{names}(values)))
end

function initialize_cpu!()
    apple =
        Sys.isapple() &&
        Sys.ARCH === :aarch64 &&
        any(lib -> occursin("Accelerate", lib.libname), BLAS.get_config().loaded_libs)
    CPU_DEFAULTS[] = cpu_defaults(apple)
    BLAS.set_num_threads(
        apple || Threads.nthreads(:default) == 1 ? min(8, Sys.CPU_THREADS) : 1,
    )
    return nothing
end
