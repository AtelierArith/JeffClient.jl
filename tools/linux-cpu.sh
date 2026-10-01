#!/usr/bin/env bash
# Sequential, equal-work CPU comparison on Linux.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CHECKPOINT="$ROOT/models/jeff-0.8b"
REFERENCE="$ROOT/examples/data/parcel_reference.json"
OUTPUT="$ROOT/artifacts/benchmarks/linux-cpu-$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLES=30
REPEATS=2
WARMUPS=10
THREADS=8
CPUS=auto
SETUP_PYTHON=0
BLAS_PROJECT=""
JULIA_BIN=${JULIA_BIN:-julia}
usage() {
    cat <<'HELP'
Usage: ./tools/linux-cpu.sh [options]
  --checkpoint DIR       Local Jeff checkpoint (default: models/jeff-0.8b)
  --reference FILE       Prepared reference JSON (default: parcel_reference.json)
  --output DIR           Empty result directory (default: timestamped artifacts path)
  --samples N            Measured forwards per process, at least 3 (default: 30)
  --repeats N            Fresh processes per implementation (default: 2)
  --warmups N            Untimed forwards, at least 2 (default: 10)
  --threads N            Shared physical-core budget (default: 8; strict single: 1)
  --cpus LIST            taskset CPU list, distinct physical cores (default: auto)
  --mkl-project DIR      Existing Julia environment containing MKL; otherwise OpenBLAS
  --setup-python         Prepare extern/jeff/.venv with uv before measurement
  --help                 Show this help
Both implementations use Float32/full sequences and return all trained scores.
Julia: N workers, BLAS 1, no interactive pool; Python: intra N, inter 1.
Processes run sequentially with alternating order and the same CPU affinity.
Setup, loading, compilation, tokenization and validation are outside warm timing.
Affinity constrains the budget; run on an otherwise idle host for stable results.
HELP
}
while (($#)); do
    case "$1" in
        --checkpoint|--reference|--output|--samples|--repeats|--warmups|--threads|--cpus|--mkl-project)
            (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --checkpoint) CHECKPOINT=$2;; --reference) REFERENCE=$2;;
                --output) OUTPUT=$2;; --samples) SAMPLES=$2;; --repeats) REPEATS=$2;;
                --warmups) WARMUPS=$2;; --threads) THREADS=$2;; --cpus) CPUS=$2;;
                --mkl-project) BLAS_PROJECT=$2;;
            esac
            shift 2;;
        --setup-python) SETUP_PYTHON=1; shift;;
        --help|-h) usage; exit 0;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
    esac
done
[[ $(uname -s) == Linux ]] || { printf 'Requires Linux.\n' >&2; exit 2; }
for name in SAMPLES REPEATS WARMUPS THREADS; do
    [[ ${!name} =~ ^[0-9]+$ ]] || { printf 'Invalid %s.\n' "$name" >&2; exit 2; }
    value=$((10#${!name}))
    printf -v "$name" '%d' "$value"
done
((SAMPLES >= 3 && REPEATS >= 1 && WARMUPS >= 2 && THREADS >= 1)) || {
    printf 'Invalid samples/repeats/warmups/threads.\n' >&2; exit 2;
}
command -v taskset >/dev/null || { printf 'Install taskset (util-linux).\n' >&2; exit 2; }
[[ -f "$CHECKPOINT/decision_config.json" && -f "$REFERENCE" ]] || {
    printf 'Checkpoint/reference missing; set --checkpoint and --reference.\n' >&2; exit 2;
}
[[ ! -d "$OUTPUT" || -z $(ls -A -- "$OUTPUT") ]] || {
    printf 'Output is not empty; choose a new --output directory.\n' >&2; exit 2;
}
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
CHECKPOINT=$(cd "$CHECKPOINT" && pwd)
if [[ -n "$BLAS_PROJECT" ]]; then
    export JEFF_BENCH_BLAS_PROJECT
    JEFF_BENCH_BLAS_PROJECT=$(cd "$BLAS_PROJECT" && pwd)
else
    unset JEFF_BENCH_BLAS_PROJECT
fi
export JULIA_PYTHONCALL_EXE=${JULIA_PYTHONCALL_EXE:-"$ROOT/extern/jeff/.venv/bin/python"}
export JULIA_CONDAPKG_BACKEND=Null
# Avoid hidden nested thread pools; implementations set their actual counts.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if ((SETUP_PYTHON)); then
    command -v uv >/dev/null || { printf 'Install uv to use --setup-python.\n' >&2; exit 2; }
    uv sync --project "$ROOT/extern/jeff" --no-default-groups > "$OUTPUT/python-setup.log" 2>&1
fi
[[ -x "$JULIA_PYTHONCALL_EXE" ]] || { printf 'Python environment missing; use --setup-python.\n' >&2; exit 2; }
printf 'Preparing dependencies and recording conditions.\n'
"$JULIA_BIN" --startup-file=no --project=tools -e 'using Pkg; Pkg.instantiate(; workspace=true)' > "$OUTPUT/julia-setup.log" 2>&1
AFFINITY=$("$JULIA_BIN" --threads=1,0 --startup-file=no --project=tools tools/linux_benchmark_config.jl "$OUTPUT" "$THREADS" "$CPUS" "$CHECKPOINT" "$REFERENCE")
{ uname -a; lscpu; cat /proc/meminfo; } > "$OUTPUT/hardware.txt"
git diff --binary > "$OUTPUT/source.patch"
run_bench() {
    local implementation=$1 run=$2 workers=$THREADS
    [[ $implementation != python ]] || workers=1
    printf 'Running %s, repeat %s, CPUs %s.\n' "$implementation" "$run" "$AFFINITY"
    if ! taskset -c "$AFFINITY" "$JULIA_BIN" --threads="$workers,0" --gcthreads=1 --startup-file=no --project=tools tools/benchmark_linux_cpu.jl "$implementation" "$CHECKPOINT" "$REFERENCE" "$SAMPLES" "$OUTPUT/$implementation-run$run.json" "$THREADS" "$WARMUPS" > "$OUTPUT/$implementation-run$run.log" 2>&1; then
        tail -50 "$OUTPUT/$implementation-run$run.log" >&2
        printf 'Failed; no summary is accepted.\n' >&2
        exit 1
    fi
}
for ((run=1; run<=REPEATS; run++)); do
    if ((run % 2)); then
        run_bench python "$run"
        run_bench julia "$run"
    else
        run_bench julia "$run"
        run_bench python "$run"
    fi
done
"$JULIA_BIN" --startup-file=no --project=tools tools/linux_benchmark_report.jl "$OUTPUT" "$SAMPLES" "$REPEATS" "$WARMUPS"
printf 'Results: %s/summary.md\n' "$OUTPUT"
