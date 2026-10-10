"""Print ONNX graph inputs/outputs (name, dtype, shape) without loading weights or running anything."""
import sys, onnx
from onnx import TensorProto
for path in sys.argv[1:]:
    m = onnx.load(path, load_external_data=False)
    print(f"== {path.split('/')[-1]}  opset={[ (o.domain or 'ai.onnx', o.version) for o in m.opset_import]}  nodes={len(m.graph.node)}")
    def dims(t):
        return [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
    for kind, vals in (("in ", m.graph.input), ("out", m.graph.output)):
        for v in vals:
            print(f"  {kind} {v.name:28s} {TensorProto.DataType.Name(v.type.tensor_type.elem_type):8s} {dims(v)}")
    ops = sorted({n.op_type for n in m.graph.node})
    print("  ops:", " ".join(ops))
