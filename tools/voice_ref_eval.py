"""Reference voice from the chosen take: does it keep the voice the same across different lines?

Compares no reference / reference only / reference + caption on the same lines and seeds.
Pitch spread across lines is a rough proxy for "same person" (smaller = steadier).
Writes out/voice/ref_<case>.wav (lines joined) and saves the speaker state for the phone.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from caption_eval import load_engine, median_f0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/voice"
TAKE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CAPTION = (ROOT / "chat/voice.txt").read_text(encoding="utf-8").strip()
ref = np.load(OUT / f"take_{TAKE}.latent.npy")

LINES = [
    "ふーん。", "は？なにそれ。", "いや、なんでだよ。", "キミ、またそれ言ってる。😒", "…ちょっとだけ、うれしいかも。",
    "今日は雨だから、外に出たくない。", "それで、結局どうなったの？", "おはよ。眠い。😪",
]
CASES = {"none+caption": (None, CAPTION), "ref": (ref, ""), "ref+caption": (ref, CAPTION)}

eng = load_engine()
for name, (r, cap) in CASES.items():
    parts, f0s = [], []
    for i, s in enumerate(LINES):
        audio, _, _ = eng.synth(s, seed=100 + i, caption=cap, ref_latent=r)
        f0s.append(median_f0(audio))
        parts += [audio, np.zeros(int(0.4 * 48000), np.float32)]
    sf.write(OUT / f"ref_{name.replace('+', '_')}.wav", np.concatenate(parts), 48000, subtype="PCM_16")
    a = np.array(f0s)
    print(f"{name:13s} pitch {np.nanmin(a):4.0f}-{np.nanmax(a):4.0f} Hz  spread(std) {np.nanstd(a):5.1f}", flush=True)

# The speaker state is constant for a fixed reference: compute it once for the phone.
t_ref = ref.shape[1] // 4 * 4
state, mask = eng.sess["speaker_encoder"].run(None, {"ref_latent": np.ascontiguousarray(ref[:, :t_ref], np.float32),
                                                     "mask": np.ones((1, t_ref), np.float32)})
np.save(OUT / f"speaker_take{TAKE}_state.npy", state)
np.save(OUT / f"speaker_take{TAKE}_mask.npy", mask)
print("speaker state", state.shape, "mask sum", float(mask.sum()), "ref frames", t_ref)
