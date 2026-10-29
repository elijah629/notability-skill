"""Offline graph construction only. Rust performs every weight update."""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnxruntime.training import artifacts
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
model = helper.make_model(helper.make_graph(
    [helper.make_node('MatMul', ['x', 'weight'], ['prediction'])],
    'rust-training-gate', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1])],
    [helper.make_tensor_value_info('prediction', TensorProto.FLOAT, [1, 1])],
    [numpy_helper.from_array(np.array([[0.0]], dtype=np.float32), 'weight')]),
    opset_imports=[helper.make_opsetid('', 18)], ir_version=9)
artifacts.generate_artifacts(model, requires_grad=['weight'],
    loss=artifacts.LossType.MSELoss, optimizer=artifacts.OptimType.SGD, artifact_directory=out, additional_output_names=["prediction"])
