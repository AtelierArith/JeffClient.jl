# Python resources used by tools are always entered through PythonCall.jl.
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
get!(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    joinpath(@__DIR__, "..", "extern", "jeff", ".venv", "bin", "python"),
)
