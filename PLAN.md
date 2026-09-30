# JeffClient implementation plan

## 1. Prepared-tensor inference — current implementation

- Load an ONNX model through the registered `ONNXRunTime` package.
- Select CPU or CUDA explicitly; keep CUDA dependencies optional.
- Define ordered choice, yes/no, and score questions.
- Accept prepared tensors and require a `(batch, options)` logits output.
- Apply the checkpoint temperature and exclude unused options before softmax.
- Reproduce Jeff's answer selection, zero-based expected score, and confidence.
- Validate this path with a tiny ONNX graph on CPU and numerical edge cases.

This milestone provides ONNX inference from prepared tensors. The exported Jeff
weights from milestone 2 are available locally under `artifacts/` (gitignored).
Inference directly from natural language still requires milestone 3.

Status: implemented. On Julia 1.13.1 / macOS arm64, the CPU ONNX graph and
answer processing and checkpoint cache passed 58 tests before the native fixture
was added. CUDA selection is exposed but has not been
validated on NVIDIA hardware.

## 2. Export a real Jeff checkpoint — implemented for fixed shapes

- Start with text-only Jeff-Qwen3.5-0.8B, including its trained readout.
- Establish whether its hybrid attention operators can be exported and executed
  by the selected ONNX Runtime version. Record unsupported operators rather
  than assuming ordinary Transformer export will work.
- Export raw logits, with temperature and option masking owned by Julia.
- Bundle decision_config.json, tokenizer assets, input/output specifications,
  model revision, and licenses with the ONNX graph and external weight files.
- Compare Python and ONNX logits for representative lengths and batch sizes.
  Export tooling may use Python; runtime inference should not require Python.

Status: exported the public Jeff-Qwen3.5-0.8B snapshot
`0f212b3e72acb4dde3f7da61e925d6ab7f819990` using `tools/export_onnx.jl`
(Python resources are accessed through PythonCall.jl).
The first graph uses float32, batch size 1, and sequence length 256. Three
English/Japanese reference cases passed in Julia with ONNX Runtime 1.20.1;
the maximum absolute logit error against PyTorch was `1.5258789e-5`.
The exporter promotes uint8 comparisons to int32 to support ORT 1.20.1.
Dynamic lengths, larger batches, CUDA, and performance tuning remain open.

### Source checkpoint cache — implemented

`resolve_checkpoint` follows the cache strategy in `../Laya.jl/src/agent.jl`:
consult the existing HF cache, then JeffClient's Scratch cache. Julia Downloads
fetches only checkpoint/tokenizer/license files, pins them by commit, stages
incomplete transfers, and publishes branch references after completeness checks.
Offline reuse and interrupted-transfer recovery are tested. A real HF download
was also exercised from Julia, reusing the already-downloaded large weight file.

## 3. Julia text preparation

- Port option descriptions and checkpoint-specific prompt layout.
- Implement tokenizer/chat-template handling against the bundled assets.
- Match token IDs, left padding, masks, and positions against reference inputs.
- Add an API accepting state and questions; retain prepared-tensor access.
- Load checkpoint temperature and max_options automatically from metadata.

## 4. Accelerator validation

- Validate the exported model on CPU first, then CUDA on suitable hardware.
- Record numerical tolerances, unsupported operators, actual provider placement,
  latency, and memory consumption. Reject silent provider changes at API level.
- Investigate additional providers separately: the current ONNXRunTime.jl
  high-level API exposes CPU and CUDA only. CoreML and AMD support must not be
  promised based solely on ONNX Runtime's provider list.

## 5. Native Julia backend

- Add a native backend implementing the same question/answer API.
- Implement the model and kernels progressively; compare with the ONNX reference.
- Extend to training and images after text inference is correct and useful.

Status: text-only Qwen3.5 implemented directly in Julia, with safetensors loading,
partial RoPE, full attention, Gated DeltaNet, RMS normalization, MLP, and readout.
Three real prepared-input cases matched PyTorch on CPU (maximum logit error
`1.7166138e-5`). Batches are evaluated one sequence at a time; tokenizer,
generation, images, training, and performance tuning remain open.

## 6. Apple GPU with Metal.jl — implemented

The chosen Apple backend is the optional `JeffClientMetalExt` extension.
Load it with `import Metal`, then `NativeBackend(checkpoint; device=:metal)`.
It uses MtlArray operations, product-only MPSGraph matmuls, and fused Julia Metal
kernels. DeltaNet uses a SIMD recurrent kernel for key widths up to 256 and a
stable chunked triangular solver otherwise. The initial finite-series inverse
was numerically unstable in trained heads.

Three English/Japanese cases matched PyTorch with maximum logit error
`9.536743e-6` on the pinned 0.8B checkpoint. This establishes correctness for
these initial cases. The expanded validation below covers the current kernels.

Expanded correctness validation on Apple M4 / Julia 1.13.1 / Metal 1.11.1:
12 cases × 3 passes, covering mixed English/Japanese prompts, batch sizes 2/3,
left padding, and lengths 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512.
Independent PyTorch float32 references were generated with PythonCall; maximum
absolute logit error for the current fused version is `3.361702e-5`. Metal scalar indexing was disabled and
GC ran between cases. The verifier uses `atol=rtol=2e-4`.

Speed baseline on Apple M4, batch 1 / length 256 / 101 active tokens, 10 warmed
calls: Julia Metal Float32 2.450 s; Julia CPU Float32 1.901 s; original Python
CPU Float32 2.988 s; original PyTorch MPS BF16 0.333 s; original PyTorch MPS
Float32 0.351 s. This initial Metal version was about 7× slower than MPS at equal precision.
The benchmark tools preserve model loading and first-call timing separately,
check reference logits, and synchronize GPU work. Profiling was collected;
performance optimization remains open. Accuracy is validated independently of
these latency measurements.

Type inspection found an abstract attention parameter in the layer vector,
propagating `hidden::Any`. A vector of the two concrete layer types removes all
nine JeffClient JET runtime-dispatch reports. Allocation and latency stayed
essentially unchanged (2,845,912 allocations, 145.8 MB Julia heap, 2.492 s after
the fix). Profile/Allocs identify repeated DeltaNet masks/intermediates and MPS
submission as allocation targets. GPU kernel timings have not been isolated.

Laya-based improvements are implemented: per-queue private buffer reuse,
MPSGraph encoding into Metal's current batch with queued lifetime roots,
shared-memory uploads recycled only after synchronization, and causal-mask
sharing across DeltaNet heads/chunks. Metal internal API use is pinned to
Metal 1.11.1. Intermediate measurements: Metal 1.586 s / 2,647,320 Julia allocations /
135.9 MB Julia heap; current CPU 1.834 s. Metal latency dropped ~36% versus
2.492 s; at that stage it remained ~4.5× slower than original Float32 PyTorch MPS. Allocation
count fell ~7%; substantial host allocation remains. The device buffer pool
retained 1.47 GB of free buffers during this benchmark, so caching is not a
claim of reduced total resident memory.

The shared-upload implementation passed all 12 cases × 3 passes with
the same maximum logit error (`3.3408403e-5`), GC between cases, and scalar GPU
indexing disabled. Latest JET reports are clean for the inspected package and
extension methods at that stage.

### Fused kernels — implemented; allocation tuning continues

- Replace DeltaNet's per-head/chunk products with one recurrent kernel. Keeping
  each value row in one SIMD group removes per-token threadgroup barriers.
- Batch full-attention head matmuls and fuse causal depthwise convolution.
- Use a product-only MPSGraph, omitting alpha/beta operations and destination
  reads. NaN-poisoned matrix/transpose/batch checks pass.
- Fuse RMS/L2 normalization and masked softmax, using Laya's SIMD reductions.
  An explicit accumulator loop avoids LLVM's crash on a 32-element tuple sum.
- Fuse Q/K RMS, partial RoPE, grouped head expansion, and the MPS head layout;
  merge output layout and sigmoid gating in one kernel. Cache the current RoPE
  table pair per queue with width/length/base in its key.
- Use Cthulhu/TypedSyntax to inspect profiled allocation sites, and reuse MPS
  shape metadata in cached graphs with explicit Objective-C retain/release.
- Retain fixed feed/result key arrays and construct their Objective-C
  dictionaries directly, avoiding Julia Dict storage and conversion copies.

Current Apple M4 result, batch 1 / length 256 / 101 active tokens, Float32,
20 warmed synchronized calls: Julia Metal **0.195 s**, original Python MPS
**0.350 s**. Julia heap: **10,513 allocations / 457,232 bytes**. This is ~12.8×
faster than the 2.492 s implementation, with ~99.6% fewer allocations. Loading
and compilation are excluded; readout and CPU score return are included.
The current implementation passes all 12 real-model cases × 3 passes, including
GC between cases and disabled scalar indexing. The tiny fixture and primitive
checks cover independent scores, real RMS widths, grouped head layouts,
partial/full/no RoPE, gates, and cache keys. All six inspected JET targets are
clean. Direct feed construction reduced allocations by another 15.4% while
median latency remained unchanged. RMS launches and residual/MLP sites account
for about 28% of the latest 10% allocation sample; buffer wrappers also remain.

Packed Q/K normalization and direct V recurrent reads remove the three QKV
slice copies. Primitive checks cover widths 7/128/256 and lengths 1/9/65.
Matched batch 2 / length 512 / F32 medians are Julia 806 ms and Python 1,339 ms.
Private cache snapshots after trial/GC/trim are 4.06/5.06/4.77 GB; this is not
peak GPU memory. Completed downloads now trim oversized free caches, while
late GC returns may exceed the limit until the next trim.
DeltaNet RMS/SiLU gate fusion and in-place residual/MLP activation are verified.
Fixed-length MPS pointer storage removes the conversion pointer Vector.
An opt-in task-local workspace now reuses intermediate arrays, their MPS
tensor-data, and result value Vectors. Dedicated embedding gather avoids GPU
index bounds checking after validating host IDs. The prepared B1/L256/F32 case
measured 6,366 allocations / 319,616 bytes / 193 ms after feed Vector reuse,
paired Q/K launch, beta/decay fusion, and dedicated mask/MLP/residual kernels, with 447 arrays retaining
857,899,008 bytes. RMS launch parameter packing reduced another 201 allocations
with unchanged bytes; layer-boundary residual/RMS fusion and readout views
removed another 471 allocations / 4,576 bytes. The preceding MLP revision also reproduced 193 ms in an
independent 20-run repeat; residual addition reduced allocation further. Real-model
12-case × 3-pass validation and GC/identity primitive checks pass.
Next work: workspace lifetime and retention control,
intermediate lifetimes, MPS feed containers, and true batch execution. Reduce
remaining allocations and measure changes against independent references.
Detailed findings
belong in `memories/MEMORY.md`, principles in `AGENTS.md`, and repeated procedures
in skills.

### MLX — alternative investigated, not implemented

MLX is a separate backend, rather than an ONNX execution provider. The official
[mlx-c](https://github.com/ml-explore/mlx-c) interface can be called with Julia
`ccall` without Python. A Julia wrapper would own MLX arrays, lazy evaluation,
stream selection, and handle lifetime. Qwen's model layers and Jeff's readout
would still need to be implemented over those operations.

MLX bindings are deferred in favor of the working Metal.jl backend above.
Metal.jl provides native Julia GPU operations; it is not an MLX binding.
