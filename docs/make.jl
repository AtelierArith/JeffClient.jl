using Documenter
using JeffClient

makedocs(;
    sitename = "JeffClient.jl",
    authors = "Satoshi Terasaki",
    modules = [JeffClient],
    remotes = nothing,
    doctest = false,
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/AtelierArith/JeffClient.jl",
        edit_link = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "Models and export" => "models.md",
        "Inference" => "inference.md",
        "Performance" => "performance.md",
        "Profiling and measurements" => "profiling.md",
        "API reference" => "api.md",
        "Development" => "development.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(;
        repo = "github.com/AtelierArith/JeffClient.jl.git",
        devbranch = "main",
        push_preview = false,
    )
end
