"""Tiny NPU experiments to separate "too many weights" from "an unsupported op" (cut1 hangs on the phone).

E1 big weights: 20 x Conv 1x1 1536->1536 (~190 MB fp32), input [1,1536,1,44]
E2 snake:       x + 1/a * sin(a*x)^2 on [1,1536,1,44] (the decoder's activation), no weights to speak of
E3 convT:       the decoder's first ConvTranspose 1536->768, kernel (1,24), stride (1,12) (~113 MB)
E4 entry:       out_proj Conv 32->1024 + model.0 Conv 1024->1536 kernel (1,7) (~44 MB), input [1,32,1,44]

All 2-D, static shapes, opset 17. Writes <out>/npu_E*.onnx.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def save(name, nodes, inp, outp, inits):
    g = helper.make_graph(nodes, name, [inp], [outp], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, str(out / f"{name}.onnx"))
    print(name, f"{(out / f'{name}.onnx').stat().st_size / 1e6:.0f} MB")


def w(shape, scale):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


C, T = 1536, 44
# E1: many large 1x1 convs
inits, nodes, x = [], [], "x"
for i in range(20):
    inits += [numpy_helper.from_array(w((C, C, 1, 1), 1 / np.sqrt(C)), f"w{i}"), numpy_helper.from_array(np.zeros(C, np.float32), f"b{i}")]
    nodes.append(helper.make_node("Conv", [x, f"w{i}", f"b{i}"], [f"y{i}"], kernel_shape=[1, 1]))
    x = f"y{i}"
save("npu_E1_bigweights", nodes, helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, 1, T]),
     helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, C, 1, T]), inits)

# E2: Snake activation exactly as exported (Mul, Sin, Pow 2, Reciprocal of alpha, Mul, Add)
alpha = numpy_helper.from_array(np.abs(w((1, C, 1, 1), 1.0)) + 0.1, "alpha")
two = numpy_helper.from_array(np.array(2.0, np.float32), "two")
nodes = [
    helper.make_node("Mul", ["x", "alpha"], ["ax"]),
    helper.make_node("Sin", ["ax"], ["s"]),
    helper.make_node("Pow", ["s", "two"], ["s2"]),
    helper.make_node("Reciprocal", ["alpha"], ["ra"]),
    helper.make_node("Mul", ["s2", "ra"], ["t"]),
    helper.make_node("Add", ["x", "t"], ["y"]),
]
save("npu_E2_snake", nodes, helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, 1, T]),
     helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, 1, T]), [alpha, two])

# E3: first upsampling ConvTranspose
inits = [numpy_helper.from_array(w((C, 768, 1, 24), 0.02), "wt"), numpy_helper.from_array(np.zeros(768, np.float32), "bt")]
nodes = [helper.make_node("ConvTranspose", ["x", "wt", "bt"], ["y"], kernel_shape=[1, 24], strides=[1, 12], pads=[0, 6, 0, 6])]
save("npu_E3_convT", nodes, helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, 1, T]),
     helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 768, 1, T * 12]), inits)

# E4: entry convs
inits = [numpy_helper.from_array(w((1024, 32, 1, 1), 0.1), "w0"), numpy_helper.from_array(np.zeros(1024, np.float32), "b0"),
         numpy_helper.from_array(w((C, 1024, 1, 7), 0.01), "w1"), numpy_helper.from_array(np.zeros(C, np.float32), "b1")]
nodes = [helper.make_node("Conv", ["x", "w0", "b0"], ["h"], kernel_shape=[1, 1]),
         helper.make_node("Conv", ["h", "w1", "b1"], ["y"], kernel_shape=[1, 7], pads=[0, 3, 0, 3])]
save("npu_E4_entry", nodes, helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, 1, T]),
     helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, 1, T]), inits)
