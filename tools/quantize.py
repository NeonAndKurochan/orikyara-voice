"""Quantize the MatMul-heavy graphs (DiT, text encoder) and save them next to the FP32 ones.

  python -I quantize.py <src v4.1-small-mf dir> <out dir>

Variants:
  int8dyn  : dynamic int8 (weights int8 per-channel, activations quantized at run time) -> MatMulInteger
  nbits4   : weight-only 4-bit blocks (MatMulNBits, block 32), activations stay FP32
  nbits8   : weight-only 8-bit blocks (MatMulNBits, block 32)
"""
import sys
import time
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
graphs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["dit_step", "text_encoder"]

for g in graphs:
    t0 = time.perf_counter()
    dst = out / f"{g}.int8dyn.onnx"
    if not dst.exists():
        quantize_dynamic(str(src / f"{g}.onnx"), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul"])
    print(f"{g} int8dyn {dst.stat().st_size / 1e6:.0f} MB  {time.perf_counter() - t0:.0f}s", flush=True)

    for bits in (4, 8):
        t0 = time.perf_counter()
        dst = out / f"{g}.nbits{bits}.onnx"
        if dst.exists():
            continue
        model = onnx.load(str(src / f"{g}.onnx"))
        cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=bits)
        q = MatMulNBitsQuantizer(model, algo_config=cfg)
        q.process()
        onnx.save_model(q.model.model, str(dst))
        print(f"{g} nbits{bits} {dst.stat().st_size / 1e6:.0f} MB  {time.perf_counter() - t0:.0f}s", flush=True)
