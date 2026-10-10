"""How often does a sentence come out low-pitched (male-sounding)? Many short chat lines, with and without caption."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from caption_eval import CASES, OUT, load_engine, median_f0  # noqa: E402

eng = load_engine()

LINES = [
    "こんばんは！😊", "こんばんは！", "やっほー！", "おはよう、よく眠れた？", "うん、そうだね。", "へえ、そうなんだ！",
    "それ、すっごく楽しそう！😆", "ちょっと待ってね。", "なるほどね〜。", "わかった、まかせて！", "えっ、ほんとに？😮",
    "ごめんね、知らなかったよ🥺", "ふふっ、どうでしょう😏", "今日もおつかれさま！", "お腹すいたね。", "いいね、行こう行こう！",
    "そっかぁ、残念。", "だいじょうぶ？", "ありがとう、うれしいな😊", "じゃあ、またあとでね！", "あはは、おもしろい！",
    "うーん、どうしよっかな。", "それでそれで？", "わたしもそう思う！",
]
LOW = 175.0
print(f"{'case':8s} low(<{LOW:.0f}Hz)  min   median  (n={len(LINES)})")
for name, cap in CASES.items():
    f0s = []
    for i, s in enumerate(LINES):
        audio, _, _ = eng.synth(s, seed=0, caption=cap)
        f0 = median_f0(audio)
        f0s.append(f0)
        if f0 < LOW:
            sf.write(OUT / f"low_{name}_{i}.wav", audio, 48000, subtype="PCM_16")
    a = np.array(f0s)
    low = [LINES[i] for i in range(len(a)) if a[i] < LOW]
    print(f"{name:8s} {len(low):2d}            {np.nanmin(a):4.0f}  {np.nanmedian(a):5.0f}   {low}", flush=True)
