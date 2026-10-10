#!/bin/sh
# GPU decoder (LiteRT) for a fixed window of W latent frames: static ONNX -> ORT basic constant
# folding (folds the Snake reciprocals) -> onnx2tf -> fereko-pack-gpu/decW_f32.tflite.
# Smaller windows make each GPU job shorter (so the 3D view stalls less while the voice is
# synthesized) at some cost in total decode time (each window carries 2 x 6 frames of context).
#   tools/make_dec_tflite.sh 36 28 ...
cd "$(dirname "$0")/.." || exit 1
for W in "$@"; do
  .venv/Scripts/python.exe -I -c "
import onnx, onnxruntime as ort
m = onnx.load('models/irodori-mf-a7542a4/v4.1-small-mf/dacvae_decode.onnx')
W = $W
d = m.graph.input[0].type.tensor_type.shape.dim; d[0].dim_value, d[1].dim_value = 1, W
o = m.graph.output[0].type.tensor_type.shape.dim; o[0].dim_value, o[2].dim_value = 1, W * 1920
onnx.save(m, 'out/tflite/dec${W}_static.onnx')
so = ort.SessionOptions(); so.log_severity_level = 3
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = 'out/tflite/dec${W}_folded.onnx'
ort.InferenceSession('out/tflite/dec${W}_static.onnx', so, providers=['CPUExecutionProvider'])
print('folded', W)
" || exit 1
  .venv-tf/Scripts/onnx2tf.exe -i "out/tflite/dec${W}_folded.onnx" -o "out/tflite/w${W}" -n > "out/tflite/convert${W}.log" 2>&1 || { echo "onnx2tf failed for $W"; exit 1; }
  cp "out/tflite/w${W}/dec${W}_folded_float32.tflite" "fereko-pack-gpu/dec${W}_f32.tflite"
  echo "done W=$W"
done
