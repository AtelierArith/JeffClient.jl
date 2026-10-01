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

Past Apple M4 performance measurements have been withdrawn. Current matched
CPU/GPU comparisons and their reproduction commands are in
[Profiling and measurements](docs/src/profiling.md).

Type inspection identified and removed an abstract attention parameter in the
layer vector. Laya-based queue ownership, pooled buffers, shared-memory uploads
and causal-mask reuse are implemented. Metal internal API use is pinned to
Metal 1.11.1. Performance claims require fresh matched measurements.

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

Packed Q/K normalization, recurrent reads, fused residual/RMS operations,
embedding gather and task-local workspace reuse are implemented. Workspace
lifetime, GC, shape changes and input ownership remain part of validation.
Experimental MLP packing is disabled by default. Historical M4 latency and
allocation comparisons have been withdrawn.

For true Metal batch execution, the current row loop in `logits` must be replaced
by a batch hook after input validation. Flatten tokens by sample for shared
projection/MLP products, then keep sample boundaries explicit in convolution and
DeltaNet recurrent state. Full attention needs `(head_dim, sequence, heads*batch)`
layouts, per-sample RoPE positions and masks, and one final readout column per
sample. Concatenating tokens into the existing attention methods would allow
cross-sample convolution or attention and is incorrect. Report actual computed
lengths when using a shared padded length; compare against both row-wise trimming
and original PyTorch at the same precision. Verify duplicated/permuted samples,
interior mask holes, GC, and workspace reuse before claiming batch speedups.
Detailed findings
belong in `memories/MEMORY.md`, principles in `AGENTS.md`, and repeated procedures
in skills.

Remaining Metal tuning work is tracked in GitHub issues:

- [#1: batch performance conditions and configuration/ownership validation](https://github.com/AtelierArith/JeffClient.jl/issues/1)
- [#2: layer buffer and tensor-data reuse to reduce GPU workspace retention](https://github.com/AtelierArith/JeffClient.jl/issues/2)
- [#3: remaining kernel submission heap allocations](https://github.com/AtelierArith/JeffClient.jl/issues/3)

### MLX — alternative investigated, not implemented

MLX is a separate backend, rather than an ONNX execution provider. The official
[mlx-c](https://github.com/ml-explore/mlx-c) interface can be called with Julia
`ccall` without Python. A Julia wrapper would own MLX arrays, lazy evaluation,
stream selection, and handle lifetime. Qwen's model layers and Jeff's readout
would still need to be implemented over those operations.

MLX bindings are deferred in favor of the working Metal.jl backend above.
Metal.jl provides native Julia GPU operations; it is not an MLX binding.
