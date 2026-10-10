"""ReazonSpeech k2 v2 (zipformer transducer) with sherpa-onnx on PC: accuracy and speed on test WAVs.

  python -I asr_check.py <wav> [<wav> ...]      (any sample rate; resampled by sherpa-onnx)
"""
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

M = Path(__file__).resolve().parent.parent / "models/reazonspeech-k2-v2"


def make(prec: str, threads: int = 4):
    sfx = ".int8" if prec == "int8" else ""
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(M / f"encoder-epoch-99-avg-1{sfx}.onnx"),
        decoder=str(M / f"decoder-epoch-99-avg-1{sfx}.onnx"),
        joiner=str(M / f"joiner-epoch-99-avg-1{sfx}.onnx"),
        tokens=str(M / "tokens.txt"),
        num_threads=threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )


for prec in ["int8"]:
    t = time.perf_counter()
    rec = make(prec)
    print(f"[{prec}] load {time.perf_counter() - t:.2f}s")
    for wav in sys.argv[1:]:
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        s = rec.create_stream()
        s.accept_waveform(sr, audio)
        t = time.perf_counter()
        rec.decode_stream(s)
        dt = time.perf_counter() - t
        print(f"  {Path(wav).name}: {len(audio) / sr:.1f}s audio, {dt * 1000:.0f} ms -> {s.result.text}")
