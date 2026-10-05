# Native Qwen fixture

`native/` contains a synthetic two-layer Qwen3.5 model (DeltaNet then full
attention), an eight-dimensional hidden state, a 64-token vocabulary, and a
three-option readout. Its float32 weights and reference logits are generated
with seed 731 using independent PyTorch/Transformers code. Cases cover lengths
1, 3, 63, 64, and 65, batches of two, left padding, and grouped heads.

Regenerate with `julia --project=tools tools/build_native_fixture.jl`, and its
CUDA mask/shape probes (`cuda_reference.json`) with
`julia --project=tools tools/build_cuda_reference.jl`; Python is reached
through PythonCall.jl. This does not download trained weights. Check a
backend against it manually with:

```bash
julia --project=tools tools/verify.jl cpu test/fixtures/native test/fixtures/native/reference.json
```
