"""Compare quantized DiT / text encoder against FP32 with the same initial noise.

Writes out/quant/<variant>_<n>.wav for listening and prints SNR of latent and waveform vs FP32,
predicted length, and stage timings (6 threads, PC CPU).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pc_synth import IrodoriOnnx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "models/irodori-mf-a7542a4/v4.1-small-mf"
Q = ROOT / "models/quant"
OUT = ROOT / "out/quant"
OUT.mkdir(parents=True, exist_ok=True)

TEXTS = [
    "こんにちは。今日はいい天気ですね。",
    "えー、ほんとに？すごいじゃん！😆",
    "ねえ、聞いて。きのうね、駅前の新しいパン屋さんに行ったんだけど、メロンパンがすっごくおいしかったの。",
]
VARIANTS = {
    "fp32": {},
    "dit_int8dyn": {"dit_step": Q / "dit_step.int8dyn.onnx"},
    "dit_nbits8": {"dit_step": Q / "dit_step.nbits8.onnx"},
    "dit_nbits4": {"dit_step": Q / "dit_step.nbits4.onnx"},
    "te_nbits8": {"text_encoder": Q / "text_encoder.nbits8.onnx"},
    "te_nbits4": {"text_encoder": Q / "text_encoder.nbits4.onnx"},
    "both_nbits8": {"dit_step": Q / "dit_step.nbits8.onnx", "text_encoder": Q / "text_encoder.nbits8.onnx"},
    "both_nbits4": {"dit_step": Q / "dit_step.nbits4.onnx", "text_encoder": Q / "text_encoder.nbits4.onnx"},
}


def snr(ref, x):
    n = min(len(ref), len(x))
    ref, x = ref[:n], x[:n]
    return 10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-20))


ref = {}
rows = []
for name, files in VARIANTS.items():
    t0 = time.perf_counter()
    eng = IrodoriOnnx(PACK, 6, files)
    load = time.perf_counter() - t0
    for i, text in enumerate(TEXTS):
        noise = ref[i]["noise"] if name != "fp32" else None
        try:
            audio, rep, ten = eng.synth(text, seed=i, noise=noise)
        except ValueError as e:  # predicted length changed, so the FP32 noise doesn't fit
            audio, rep, ten = eng.synth(text, seed=i)
            rep["note"] = f"length changed: {e}"
        sf.write(OUT / f"{name}_{i}.wav", audio, 48000, subtype="PCM_16")
        if name == "fp32":
            ref[i] = {"noise": ten["noise"], "latent": ten["latent"], "audio": audio, "frames": rep["latent_frames"]}
            lat_snr = aud_snr = float("inf")
        else:
            same = ten["latent"].shape == ref[i]["latent"].shape
            lat_snr = snr(ref[i]["latent"].ravel(), ten["latent"].ravel()) if same else float("nan")
            aud_snr = snr(ref[i]["audio"], audio) if same else float("nan")
        t = rep["timings_ms"]
        rows.append((name, i, rep["latent_frames"], ref[i]["frames"], lat_snr, aud_snr,
                     t.get("text_encoder", 0), t.get("dit_step", 0), rep.get("note", "")))
    print(f"{name}: load {load:.1f}s", flush=True)
    del eng

print(f"\n{'variant':12s} # frames(fp32)  latentSNR  audioSNR  text_enc_ms  dit_ms")
for name, i, fr, fr0, ls, a, te, dit, note in rows:
    print(f"{name:12s} {i} {fr:4d}({fr0:4d})  {ls:8.1f}  {a:8.1f}  {te:10.0f}  {dit:7.0f} {note}")
