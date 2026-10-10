"""Rewrite every 1-D Conv / ConvTranspose as a 2-D one with height 1 (Unsqueeze -> Conv2d -> Squeeze).

NPUs (QNN HTP) handle 2-D convolutions natively; on the phone the QNN EP rejected a 1-D conv during
its NHWC layout transform. The rewrite is exact: weights [O, I, K] -> [O, I, 1, K], attributes padded
with a leading 1/0 for the new height axis.

  python -I conv1d_to_2d.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
inits = {i.name: i for i in g.initializer}

axes = numpy_helper.from_array(np.array([2], np.int64), "c2d_axes_h")
g.initializer.append(axes)

new_nodes = []
count = 0
for n in g.node:
    if n.op_type not in ("Conv", "ConvTranspose"):
        new_nodes.append(n)
        continue
    w = inits[n.input[1]]
    if len(w.dims) != 3:
        new_nodes.append(n)
        continue
    count += 1
    arr = numpy_helper.to_array(w)
    w.CopyFrom(numpy_helper.from_array(arr[:, :, None, :], w.name))

    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    new_attrs = {}
    for k, v in attrs.items():
        if k in ("kernel_shape", "strides", "dilations"):
            new_attrs[k] = [1] + list(v)
        elif k == "pads":                       # [begin, end] -> [h_begin, w_begin, h_end, w_end]
            new_attrs[k] = [0, v[0], 0, v[1]]
        elif k == "output_padding":
            new_attrs[k] = [0] + list(v)
        elif k == "output_shape":
            new_attrs[k] = [1] + list(v)
        else:
            new_attrs[k] = v

    x4 = n.input[0] + f"__h{count}"
    y4 = n.output[0] + "__h"
    new_nodes.append(helper.make_node("Unsqueeze", [n.input[0], axes.name], [x4], name=n.name + "/c2d_unsq"))
    inputs = [x4, n.input[1]] + list(n.input[2:])
    new_nodes.append(helper.make_node(n.op_type, inputs, [y4], name=n.name + "/c2d", **new_attrs))
    new_nodes.append(helper.make_node("Squeeze", [y4, axes.name], [n.output[0]], name=n.name + "/c2d_sq"))

del g.node[:]
g.node.extend(new_nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)
print(f"rewrote {count} convolutions -> {dst}")
