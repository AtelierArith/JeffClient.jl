# Development

Julia 1.13 is the development baseline. CPU tests use a tiny ONNX graph and a small synthetic Qwen fixture with independent PyTorch reference scores. They do not download Jeff's weights or require Python.

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Continuous integration

The [CI workflow](https://github.com/AtelierArith/JeffClient.jl/blob/main/.github/workflows/CI.yml)
was generated with PkgTemplates.jl 0.7.63's `GitHubActions` plugin. It runs on
pushes to `main`, tags, pull requests, and manual dispatch, using Julia 1.13 on
Ubuntu x64. Julia's build and test actions run the package's existing CPU fixture
tests. Metal GPU validation remains a separate hardware-dependent workflow.
Coverage uploads are disabled, so no coverage-service token is required.

## Build the documentation

Run these commands from the repository root:

```bash
julia --project=docs -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(; workspace=true)'
julia --project=docs docs/make.jl
```

Generated HTML is written to `docs/build/` and excluded from Git. Serve that directory over HTTP to preview it. The documentation workflow builds pull requests and publishes main to the `gh-pages` branch; configure GitHub Pages to serve that branch.
