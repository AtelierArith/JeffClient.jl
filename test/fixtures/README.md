# ONNX fixture

`logits.onnx` is committed in this repository. A normal Git clone includes it;
running the demo or `Pkg.instantiate()` does not generate it. Regeneration is
optional and is not part of the initial setup.

`logits.onnx` is a tiny Identity graph: float tensors named `scores` and
`logits`, with dynamic `(batch, options)` dimensions. It contains no weights.
It uses ONNX IR version 8 and opset 13. The test suite executes this graph with
the actual CPU runtime to check dimension conventions and postprocessing.

Regenerate it using `python3 test/fixtures/build_fixture.py`. The generator
encodes the small protobuf directly and needs no third-party dependencies.
Python is not needed to run the Julia package or its tests.

## Native Qwen fixture

`native/` contains a synthetic two-layer Qwen3.5 model (DeltaNet then full
attention), an eight-dimensional hidden state, a 64-token vocabulary, and a
three-option readout. Its float32 weights and reference logits are generated
with seed 731 using independent PyTorch/Transformers code. Cases cover lengths
1, 3, 63, 64, and 65, batches of two, left padding, and grouped heads.

Regenerate with `julia --project=tools tools/build_native_fixture.jl`; Python
resources are accessed through PythonCall.jl. This does not download trained
weights. Compare a backend manually with:

```bash
julia --project=tools tools/verify_native.jl test/fixtures/native cpu test/fixtures/native/reference.json
julia --project=tools tools/verify_native.jl test/fixtures/native metal test/fixtures/native/reference.json
```
