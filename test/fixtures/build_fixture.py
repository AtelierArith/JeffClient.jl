"""Generate a minimal ONNX Identity graph without external dependencies."""
from pathlib import Path


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def integer(field, value):
    return varint(field << 3) + varint(value)


def message(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((field << 3) | 2) + varint(len(value)) + value


def tensor_info(name):
    shape = message(1, message(2, "batch")) + message(1, message(2, "options"))
    tensor_type = integer(1, 1) + message(2, shape)  # TensorProto.FLOAT
    return message(1, name) + message(2, message(1, tensor_type))


node = message(1, "scores") + message(2, "logits") + message(4, "Identity")
graph = (message(1, node) + message(2, "jeffclient_logits_fixture")
         + message(11, tensor_info("scores")) + message(12, tensor_info("logits")))
model = (integer(1, 8) + message(2, "JeffClient fixture generator")
         + message(7, graph) + message(8, integer(2, 13)))
Path(__file__).with_name("logits.onnx").write_bytes(model)
