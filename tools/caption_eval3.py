"""Pick a voice caption for the downer/cool persona: pitch per line (to rule out male-sounding output)."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from caption_eval import OUT, load_engine, median_f0  # noqa: E402

eng = load_engine()
LINES = [
    "ふーん。", "は？", "…で？", "まあね。", "いや、なんでだよ。", "それはない。", "別に、いいけど。",
    "おいら、またそれ言ってる。😒", "はいはい、すごいすごい。🙄", "…ちょっとだけ、うれしいかも。", "眠い。😪",
    "それ、本気で言ってる？😏", "知らない。", "で、今日はなにしてたの？",
]
CASES = {
    "cool_a": "落ち着いた低めのトーンの若い女性。ダウナーで淡々とした、そっけない話し方。",
    "cool_b": "クールでけだるげな若い女性の声。感情をあまり出さず、ぼそっと話す。",
    "young_f": "明るくかわいい、若い女性の声",
}
LOW = 175.0
for name, cap in CASES.items():
    f0s = []
    for i, s in enumerate(LINES):
        audio, _, _ = eng.synth(s, seed=0, caption=cap)
        sf.write(OUT / f"cool_{name}_{i}.wav", audio, 48000, subtype="PCM_16")
        f0s.append(median_f0(audio))
    a = np.array(f0s)
    low = [LINES[i] for i in range(len(a)) if a[i] < LOW]
    print(f"{name:8s} low {len(low)}/{len(LINES)}  min {np.nanmin(a):4.0f}  median {np.nanmedian(a):4.0f}  {low}", flush=True)
