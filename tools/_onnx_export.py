"""Export a text-only Jeff Qwen checkpoint; Python is only used at export time.

Private helper imported through PythonCall by tools/export_onnx.jl.
The first export deliberately uses fixed batch and sequence dimensions. Inputs
must be left-padded to those dimensions, never truncated.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from jeff.models import load_decision_model


class TextLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # The text branch is identical for inputs without images. Omitting the
        # vision branch removes unused weights and multimodal position logic.
        self.backbone = model.backbone.language_model
        self.readout = model.readout

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        ).last_hidden_state[:, -1]
        return self.readout(hidden).float()


def make_ort120_compatible(graph):
    """Keep exact integer comparisons executable with Julia's ORT 1.20.1.

    PyTorch emits uint8 index comparisons for the chunk scan. Equal on uint8
    is legal ONNX but has no CPU kernel in ORT 1.20.1; casting both inputs to
    int32 preserves every uint8/int8 value exactly.
    """
    types = {v.name: v.type.tensor_type.elem_type for v in graph.graph.value_info}
    types.update({v.name: v.type.tensor_type.elem_type for v in graph.graph.input})
    types.update({v.name: v.data_type for v in graph.graph.initializer})
    nodes = []
    for node in graph.graph.node:
        if node.op_type == "CastLike" and len(node.input) == 2 and node.input[1] in types:
            node = onnx.helper.make_node("Cast", [node.input[0]], list(node.output),
                                         name=node.name, to=types[node.input[1]])
        if node.op_type == "Equal" and any(types.get(i) in (onnx.TensorProto.UINT8, onnx.TensorProto.INT8)
                                           for i in node.input):
            for index, name in enumerate(list(node.input)):
                promoted = f"{node.name}_int32_{index}"
                nodes.append(onnx.helper.make_node("Cast", [name], [promoted],
                                                  name=promoted, to=onnx.TensorProto.INT32))
                node.input[index] = promoted
        nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    graph.ir_version = 10


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-revision", required=True,
                        help="Immutable Hugging Face snapshot revision")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    if args.sequence_length < 1 or args.batch_size < 1:
        parser.error("Batch size and sequence length must be positive.")
    if len(args.source_revision) != 40 or any(c not in "0123456789abcdef" for c in args.source_revision):
        parser.error("Use a 40-character lowercase hexadecimal source revision.")

    config = json.loads((args.checkpoint / "decision_config.json").read_text())
    if config.get("architecture", "qwen") != "qwen":
        parser.error("This first exporter supports Qwen checkpoints only.")
    torch.set_num_threads(8)
    print("Loading Jeff checkpoint on CPU (float32)", flush=True)
    model = load_decision_model(checkpoint=str(args.checkpoint), device="cpu")
    model.eval()
    wrapper = TextLogits(model).eval()

    cases = [
        {"state": "The parcel arrived crushed and the customer wants a replacement.",
         "question": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"refund": "Refunds and payments", "delivery": "Damaged parcels"}}},
        {"state": "荷物が破損して届きました。交換をお願いします。",
         "question": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"refund": "Refunds and payments", "delivery": "Damaged parcels"}}},
        {"state": "The customer says everything was perfect and thanks the team.",
         "question": {"type": "noul", "instructions": "Is the customer angry?"}},
    ]
    tokenizer = model.processor.tokenizer
    samples = []
    with torch.inference_mode():
        for case in cases:
            rows = [case] * args.batch_size
            prepared = model.prepare(rows)
            original_ids = prepared.inputs["input_ids"]
            original_mask = prepared.inputs["attention_mask"]
            length = original_ids.shape[1]
            if length > args.sequence_length:
                parser.error(f"Reference prompt uses {length} tokens; increase --sequence-length.")
            padding = args.sequence_length - length
            ids = torch.nn.functional.pad(original_ids, (padding, 0), value=tokenizer.pad_token_id)
            mask = torch.nn.functional.pad(original_mask, (padding, 0), value=0)
            expected = wrapper(ids, mask)
            # Verify that left padding/text-branch extraction preserves Jeff's
            # original active scores before using the wrapper as the reference.
            original = model(prepared)
            count = prepared.counts[0]
            torch.testing.assert_close(expected[:, :count], original[:, :count], atol=2e-3, rtol=2e-3)
            samples.append({"state": case["state"], "question": case["question"],
                            "inputs": {"input_ids": ids.tolist(), "attention_mask": mask.tolist()},
                            "logits": expected.tolist()})

    args.output.mkdir(parents=True)
    graph_path = args.output / "model.onnx"
    inputs = tuple(torch.tensor(samples[0]["inputs"][name], dtype=torch.int64)
                   for name in ("input_ids", "attention_mask"))
    print("Exporting fixed-shape raw logits", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, inputs, str(graph_path), input_names=["input_ids", "attention_mask"],
            output_names=["logits"], opset_version=18, dynamo=True,
            external_data=True, optimize=False, report=True, artifacts_dir=str(args.output),
        )

    graph = onnx.load(str(graph_path), load_external_data=False)
    # This graph uses tensor/graph features supported by IR 10. ONNXRunTime.jl
    # 1.3.4 bundles ORT 1.20.1, whose maximum supported IR version is 10.
    make_ort120_compatible(graph)
    onnx.save(graph, str(graph_path))
    onnx.checker.check_model(str(graph_path))
    print("Validating three reference cases in ONNX Runtime", flush=True)
    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    max_error = 0.0
    for sample in samples:
        inputs = {name: np.asarray(value, dtype=np.int64) for name, value in sample["inputs"].items()}
        actual = session.run(["logits"], inputs)[0]
        expected = np.asarray(sample["logits"], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))

    for name in ("decision_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "LICENSE", "NOTICE"):
        source = args.checkpoint / name
        if source.is_file():
            shutil.copy2(source, args.output / name)
    metadata = {"format_version": 1, "model_file": "model.onnx", "output_name": "logits",
                "source_revision": args.source_revision, "backend": "onnx", "modality": "text",
                "batch_size": args.batch_size, "sequence_length": args.sequence_length,
                "input_names": ["input_ids", "attention_mask"], "opset": 18, "ir_version": 10,
                "torch_version": torch.__version__, "python_ort_version": ort.__version__,
                "max_absolute_error": max_error}
    (args.output / "export_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "reference.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    print(f"Export complete: {args.output}; max absolute logit error {max_error:.6g}", flush=True)
