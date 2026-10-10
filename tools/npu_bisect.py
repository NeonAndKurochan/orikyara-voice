"""Bisect the NPU hang: decoder cut after each upsampling stage, plus the full decoder on a 16-frame window.

Writes <out>/dacvae_cut{1,2,3}.onnx and dacvae_w16.onnx (all with 2-D convs, static shapes).
Compile them for HTP v73 with tools/qnn_compile_check.py (HTP_ARCH=73).

  python -I npu_bisect.py <static44c2d.onnx> <original dacvae_decode.onnx> <out dir>
"""
import subprocess
import sys
from pathlib import Path

import onnx
from onnx.utils import Extractor

c2d44, original, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

cuts = {
    "dacvae_cut1": "/decoder/model.1/block.8/Add_output_0",   # [1, 768, 528]
    "dacvae_cut2": "/decoder/model.2/block.8/Add_output_0",   # [1, 384, 5280]
    "dacvae_cut3": "/decoder/model.3/block.8/Add_output_0",   # [1, 192, 42240]
}
model = onnx.shape_inference.infer_shapes(onnx.load(str(c2d44)))
ex = Extractor(model)
for name, tensor in cuts.items():
    sub = ex.extract_model(["latent"], [tensor])
    onnx.save(sub, str(out / f"{name}.onnx"))
    print(name, "->", tensor, f"{(out / f'{name}.onnx').stat().st_size / 1e6:.0f} MB")

# Full decoder with a 16-frame window: static input/output, then the same 2-D conv rewrite.
W = 16
m = onnx.load(str(original))
d = m.graph.input[0].type.tensor_type.shape.dim
d[0].dim_value, d[1].dim_value = 1, W
o = m.graph.output[0].type.tensor_type.shape.dim
o[0].dim_value, o[2].dim_value = 1, W * 1920
tmp = out / "dacvae_w16.static.onnx"
onnx.save(m, str(tmp))
subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "conv1d_to_2d.py"), str(tmp), str(out / "dacvae_w16.onnx")], check=True)
tmp.unlink()
