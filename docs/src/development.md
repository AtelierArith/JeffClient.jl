# Development

Julia 1.13 is the development baseline. CPU tests use a tiny ONNX graph and a small synthetic Qwen fixture with independent PyTorch reference scores. They do not download Jeff's weights or require Python.

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Build the documentation

Run these commands from the repository root:

```bash
julia --project=docs -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(; workspace=true)'
julia --project=docs docs/make.jl
```

Generated HTML is written to `docs/build/` and excluded from Git. Serve that directory over HTTP to preview it. The documentation workflow builds pull requests and publishes main to the `gh-pages` branch; configure GitHub Pages to serve that branch.
