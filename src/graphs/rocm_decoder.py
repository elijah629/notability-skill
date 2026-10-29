"""Create inference graph from Rust-updated weights. Performs no training."""
import json
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
weights = json.loads(Path(sys.argv[1]).read_text())
size = weights['size']
values = np.asarray(weights['values'], dtype=np.float32).reshape(size, size)
model = helper.make_model(helper.make_graph(
    [helper.make_node('MatMul', ['x', 'weight'], ['prediction'])],
    'rust-rocm-updated-decoder',
    [helper.make_tensor_value_info('x', TensorProto.FLOAT, [size, size])],
    [helper.make_tensor_value_info('prediction', TensorProto.FLOAT, [size, size])],
    [numpy_helper.from_array(values, 'weight')]),
    opset_imports=[helper.make_opsetid('', 18)], ir_version=9)
onnx.checker.check_model(model)
onnx.save(model, sys.argv[2])
