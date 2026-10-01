#!/usr/bin/env bash
# Sequential, equal-work CPU/GPU comparison on Apple Silicon.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CHECKPOINT="$ROOT/models/jeff-0.8b"
REFERENCE="$ROOT/examples/data/parcel_reference.json"
OUTPUT="$ROOT/artifacts/benchmarks/mac-M-series-$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLES=30
REPEATS=2
AUTO_CPU=0
MLX=0
SETUP_PYTHON=0
JULIA_BIN=${JULIA_BIN:-julia}
usage() {
    cat <<'HELP'
Usage: ./tools/mac-M-series.sh [options]
  --checkpoint DIR       Local Jeff checkpoint (default: models/jeff-0.8b)
  --reference FILE       Prepared reference JSON (default: parcel_reference.json)
  --output DIR           New result directory (default: timestamped artifacts path)
  --samples N            Warmed forwards per process, at least 3 (default: 30)
  --repeats N            Fresh processes per implementation (default: 2)
  --include-auto-cpu     Add PyTorch 8 vs Julia 8 / Accelerate automatic; separate budget
  --mlx                  Add adapted MLX GPU and framework-managed CPU results
  --setup-python         Prepare extern/jeff/.venv with uv before measurement
  --help                 Show this help
Default: CPU 1 vs 1, and PyTorch MPS Float32 vs Julia Metal.jl on the same GPU.
All run full sequences; GPU timing includes CPU input upload and score download.
Setup, imports, loading, tokenization and compilation are outside warm timings.
Run from an otherwise idle machine on AC power. No CPU affinity is imposed.
HELP
}
while (($#)); do
    case "$1" in
        --checkpoint|--reference|--output|--samples|--repeats)
            (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --checkpoint) CHECKPOINT=$2;; --reference) REFERENCE=$2;;
                --output) OUTPUT=$2;; --samples) SAMPLES=$2;; --repeats) REPEATS=$2;;
            esac
            shift 2;;
        --include-auto-cpu) AUTO_CPU=1; shift;;
        --mlx) MLX=1; shift;;
        --setup-python) SETUP_PYTHON=1; shift;;
        --help|-h) usage; exit 0;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2;;
    esac
done
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || {
    printf 'Requires native arm64 macOS on Apple Silicon.\n' >&2; exit 2;
}
if [[ ! $SAMPLES =~ ^[0-9]+$ || ! $REPEATS =~ ^[0-9]+$ ]]; then
    printf 'Invalid samples/repeats.\n' >&2; exit 2
fi
SAMPLES=$((10#$SAMPLES))
REPEATS=$((10#$REPEATS))
if ((SAMPLES < 3 || REPEATS < 1)); then
    printf 'Invalid samples/repeats.\n' >&2; exit 2
fi
[[ -f "$CHECKPOINT/decision_config.json" && -f "$REFERENCE" ]] || {
    printf 'Checkpoint/reference missing; set --checkpoint and --reference.\n' >&2; exit 2;
}
mkdir -p "$OUTPUT"
[[ ! -e "$OUTPUT/summary.json" && ! -e "$OUTPUT/runtime.json" ]] || {
    printf 'Output already contains results; choose a new --output directory.\n' >&2; exit 2;
}
OUTPUT=$(cd "$OUTPUT" && pwd)
export JULIA_PYTHONCALL_EXE=${JULIA_PYTHONCALL_EXE:-"$ROOT/extern/jeff/.venv/bin/python"}
export JULIA_CONDAPKG_BACKEND=Null
if ((SETUP_PYTHON)); then
    command -v uv >/dev/null || { printf 'Install uv to use --setup-python.\n' >&2; exit 2; }
    uv sync --project "$ROOT/extern/jeff" --no-default-groups > "$OUTPUT/python-setup.log" 2>&1
    if ((MLX)); then
        uv pip install --python "$JULIA_PYTHONCALL_EXE" 'mlx-lm==0.31.3' 'mlx==0.32.3' >> "$OUTPUT/python-setup.log" 2>&1
    fi
fi
[[ -x "$JULIA_PYTHONCALL_EXE" ]] || {
    printf 'Python environment missing; use --setup-python (uv required).\n' >&2; exit 2;
}
printf 'Preparing Julia dependencies; no benchmark is running yet.\n'
"$JULIA_BIN" --startup-file=no --project=tools -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(; workspace=true)' > "$OUTPUT/julia-setup.log" 2>&1
# Capture libraries and host before any timed inference.
"$JULIA_BIN" --threads=1 --startup-file=no --project=tools tools/mac_benchmark_runtime.jl "$OUTPUT/runtime.json" > "$OUTPUT/runtime.log" 2>&1
{ uname -a; sysctl hw.physicalcpu hw.logicalcpu hw.memsize machdep.cpu.brand_string; pmset -g batt; } > "$OUTPUT/hardware.txt"
git rev-parse HEAD > "$OUTPUT/source-commit.txt"
git status --short > "$OUTPUT/worktree-status.txt"
shasum -a 256 "$CHECKPOINT"/*.safetensors "$CHECKPOINT/config.json" "$CHECKPOINT/decision_config.json" "$REFERENCE" tools/benchmark*.jl tools/mac_benchmark_*.jl tools/mac-M-series.sh > "$OUTPUT/hashes.txt"
run_bench() {
    local label=$1; shift
    printf '%s\n' "$label"
    if ! "$@" > "$OUTPUT/$label.log" 2>&1; then
        tail -50 "$OUTPUT/$label.log" >&2
        printf 'Failed: %s; no summary is accepted.\n' "$label" >&2
        exit 1
    fi
}
for ((run=1; run<=REPEATS; run++)); do
    # The previous process exits before the next starts. Independent warm-up in each.
    run_bench "cpu-single-python-run$run" "$JULIA_BIN" --threads=1 --startup-file=no --project=tools tools/benchmark_cpu_single_python.jl "$CHECKPOINT" cpu "$REFERENCE" "$SAMPLES" "$OUTPUT/cpu-single-python-run$run.json"
    run_bench "cpu-single-julia-run$run" "$JULIA_BIN" --threads=1 --startup-file=no --project=tools tools/benchmark_cpu_single_julia.jl "$CHECKPOINT" cpu "$REFERENCE" 1 "$SAMPLES" "$OUTPUT/cpu-single-julia-run$run.json"
    run_bench "gpu-pytorch-run$run" "$JULIA_BIN" --threads=8 --startup-file=no --project=tools tools/benchmark_gpu_python.jl "$CHECKPOINT" mps-f32 "$REFERENCE" "$SAMPLES" "$OUTPUT/gpu-pytorch-run$run.json"
    run_bench "gpu-metal-run$run" "$JULIA_BIN" --threads=8 --startup-file=no --project=tools tools/benchmark_metal_reference.jl "$CHECKPOINT" metal "$REFERENCE" 1 "$SAMPLES" "$OUTPUT/gpu-metal-run$run.json"
    if ((AUTO_CPU)); then
        run_bench "cpu-eight-python-run$run" "$JULIA_BIN" --threads=8 --startup-file=no --project=tools tools/benchmark_original.jl "$CHECKPOINT" cpu "$REFERENCE" "$SAMPLES" "$OUTPUT/cpu-eight-python-run$run.json"
        run_bench "cpu-auto-julia-run$run" "$JULIA_BIN" --threads=8 --startup-file=no --project=tools tools/benchmark_inference.jl "$CHECKPOINT" cpu "$REFERENCE" 1 "$SAMPLES" "$OUTPUT/cpu-auto-julia-run$run.json" --python-reference
    fi
    if ((MLX)); then
        run_bench "gpu-mlx-run$run" "$JULIA_BIN" --threads=1 --startup-file=no --project=tools tools/benchmark_mlx.jl "$CHECKPOINT" gpu "$REFERENCE" "$OUTPUT/gpu-mlx-run$run.json" "$SAMPLES" reference-norm
        run_bench "cpu-mlx-run$run" "$JULIA_BIN" --threads=1 --startup-file=no --project=tools tools/benchmark_mlx.jl "$CHECKPOINT" cpu "$REFERENCE" "$OUTPUT/cpu-mlx-run$run.json" "$SAMPLES" reference-norm
    fi
done
"$JULIA_BIN" --startup-file=no --project=tools tools/mac_benchmark_report.jl "$OUTPUT" "$SAMPLES" "$REPEATS"
printf 'Results: %s/summary.md\n' "$OUTPUT"
