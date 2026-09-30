# Reuse Jeff's Python environment without installing another Python runtime.
include("python_env.jl")
using PythonCall

sys = pyimport("sys")
sys.path.insert(0, @__DIR__)
exporter = pyimport("_onnx_export")
try
    exporter.main(pylist(ARGS))
catch exception
    if exception isa PyException && pyisinstance(exception.v, pybuiltins.SystemExit)
        exit(pyconvert(Int, exception.v.code))
    end
    rethrow()
end
