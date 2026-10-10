"""Voice caption candidates for the downer/cool ふぇれ子: same 4 lines per caption, one file per caption.

Writes out/caption/variant_<n>.wav (lines separated by short silences) and prints the pitch range.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from caption_eval import OUT, load_engine, median_f0  # noqa: E402

LINES = ["ふーん。", "いや、なんでだよ。", "キミ、またそれ言ってる。😒", "…ちょっとだけ、うれしいかも。"]
VARIANTS = [
    "低めでハスキーな若い女性。気だるそうに、小さめの声でぼそぼそ話す。",
    "10代後半くらいの少女。無表情で、平坦なイントネーションで淡々と話す。",
    "落ち着いたアルトの女性の声。クールで、少し冷たい印象。",
    "眠そうで、ため息まじりに話す若い女性。",
    "少し鼻にかかった、やる気のない若い女性の声。",
    "ささやくような小さな声で、そっけなく話す女の子。",
]

eng = load_engine()
for n, cap in enumerate(VARIANTS, 1):
    parts, f0s = [], []
    for s in LINES:
        audio, _, _ = eng.synth(s, seed=0, caption=cap)
        f0s.append(median_f0(audio))
        parts += [audio, np.zeros(int(0.4 * 48000), np.float32)]
    sf.write(OUT / f"variant_{n}.wav", np.concatenate(parts), 48000, subtype="PCM_16")
    print(f"{n}: pitch {np.nanmin(f0s):.0f}-{np.nanmax(f0s):.0f} Hz  {cap}", flush=True)
