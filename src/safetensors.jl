# Safetensors tensors use row-major storage. Reverse axes when loading so
# Julia's column-major arrays share the same feature/token ordering.
function read_native_weights(path; select = Returns(true), convert_array = identity)
    tensors = Dict{String,Any}()
    open(path, "r") do io
        header_length = Int(ltoh(read(io, UInt64)))
        0 < header_length <= filesize(path) - 8 ||
            throw(ArgumentError("Invalid safetensors header."))
        header = JSON.parse(String(read(io, header_length)))
        base = 8 + header_length
        for (name, spec) in header
            name == "__metadata__" && continue
            select(name) || continue
            shape = Tuple(reverse(Int.(spec["shape"])))
            start, stop = Int.(spec["data_offsets"])
            0 <= start <= stop <= filesize(path) - base ||
                throw(ArgumentError("Invalid offsets for tensor $name."))
            dtype = spec["dtype"]
            T =
                dtype == "BF16" ? UInt16 :
                dtype == "F16" ? Float16 :
                dtype == "F32" ? Float32 :
                throw(ArgumentError("Native backend does not support tensor dtype $dtype."))
            stop - start == prod(shape) * sizeof(T) ||
                throw(ArgumentError("Invalid size for tensor $name."))
            seek(io, base + start)
            raw = read!(io, Vector{T}(undef, prod(shape)))
            values =
                dtype == "BF16" ? reinterpret(Float32, UInt32.(ltoh.(raw)) .<< 16) :
                Float32.(ltoh.(raw))
            tensors[String(name)] = convert_array(reshape(values, shape))
        end
    end
    return tensors
end
