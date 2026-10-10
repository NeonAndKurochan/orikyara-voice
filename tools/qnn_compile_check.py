"""Offline check on PC: can the QNN HTP backend take (and compile) every node of a model?

Uses the onnxruntime-qnn plugin EP (.venv-qnn) with the HTP backend in offline mode (EP context
generation for an HTP v79 device). Prints how many nodes QNN takes, and any error.

  .venv-qnn\\Scripts\\python.exe -I tools\\qnn_compile_check.py <model.onnx> [more.onnx ...]
"""
import os
import sys
import time
from pathlib import Path

import onnxruntime as ort
import onnxruntime_qnn as oq

ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == oq.get_ep_name()]
print("QNN EP devices:", [(d.device.type, d.device.vendor, dict(d.device.metadata)) for d in devices])

for path in sys.argv[1:]:
    out = Path(path).with_suffix(f".v{os.environ.get('HTP_ARCH', '79')}.qnnctx.onnx")
    so = ort.SessionOptions()
    so.log_severity_level = int(os.environ.get("ORT_LOG", "2"))
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", str(out))
    so.add_session_config_entry("ep.context_embed_mode", "0")
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")   # fail if any node would stay on the CPU
    opts = {"backend_path": oq.get_qnn_htp_path(), "htp_arch": os.environ.get("HTP_ARCH", "79"), "enable_htp_fp16_precision": "1",
            "htp_graph_finalization_optimization_mode": "3"}
    so.add_provider_for_devices(devices, opts)
    t = time.perf_counter()
    try:
        ort.InferenceSession(path, so)
        print(f"OK  {Path(path).name}: compiled for HTP in {time.perf_counter() - t:.1f}s -> {out.name}")
    except Exception as e:  # noqa: BLE001
        print(f"NG  {Path(path).name}: {str(e)[:600]}")
