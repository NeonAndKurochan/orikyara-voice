"""Speaker similarity to the chosen take, using the model's own speaker encoder.

For each output in out/voice/ref_<case>.wav's lines, re-encode the audio (DAC-VAE encoder), take the
speaker encoder's global token (index 0 = masked mean), and report the cosine similarity to the
reference take's token. Higher and tighter = the same voice.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from caption_eval import load_engine  # noqa: E402
from voice_ref_eval import CASES, LINES, OUT, TAKE  # noqa: E402  (imports run the eval: reuse its outputs)

import onnxruntime as ort  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
enc = ort.InferenceSession(str(ROOT / "models/irodori-mf-a7542a4/v4.1-small-mf/dacvae_encode.onnx"),
                           providers=["CPUExecutionProvider"])
eng = load_engine()


def speaker_token(latent: np.ndarray) -> np.ndarray:
    t = latent.shape[1] // 4 * 4
    state, _ = eng.sess["speaker_encoder"].run(None, {"ref_latent": np.ascontiguousarray(latent[:, :t], np.float32),
                                                      "mask": np.ones((1, t), np.float32)})
    v = state[0, 0]
    return v / np.linalg.norm(v)


def token_from_audio(audio: np.ndarray) -> np.ndarray:
    x = audio.astype(np.float32)
    x = x / max(1e-6, np.sqrt(np.mean(x ** 2))) * 0.1   # rough level match (upstream uses -16 LUFS)
    (lat,) = enc.run(None, {"waveform": x[None, None, :]})
    return speaker_token(lat)


ref_tok = speaker_token(np.load(OUT / f"take_{TAKE}.latent.npy"))
for name, (r, cap) in CASES.items():
    sims = []
    for i, s in enumerate(LINES):
        audio, _, _ = eng.synth(s, seed=100 + i, caption=cap, ref_latent=r)
        sims.append(float(token_from_audio(audio) @ ref_tok))
    a = np.array(sims)
    print(f"{name:13s} similarity to take {TAKE}: mean {a.mean():.3f}  min {a.min():.3f}  " + " ".join(f"{v:.2f}" for v in a), flush=True)
