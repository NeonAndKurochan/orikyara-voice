"""Does a voice caption keep the voice female and steady? Measures median F0 per sentence.

Writes out/caption/<case>_<n>.wav and prints the median pitch (Hz) of each sentence.
Rough guide: adult male speech ~85-155 Hz, adult female ~165-255 Hz, bright/young voices higher.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pc_synth import IrodoriOnnx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/caption"
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    "こんばんは！", "はじめまして、ふぇれ子だよ。", "えー、ほんとに？すごいじゃん！😆", "今日はなにしてたの？",
    "うーん、それはちょっと難しいかも。", "ねえねえ、聞いて聞いて！", "おやすみ、また明日ね😊", "そっか、大変だったね。",
]
CASES = {
    "none": "",
    "young_f": "明るくかわいい、若い女性の声",
    "calm_f": "落ち着いた、近い距離感の若い女性話者。親しい相手に楽しそうに話している。",
}


def median_f0(x: np.ndarray, sr: int = 48000) -> float:
    """Autocorrelation pitch on voiced 40 ms frames; median over frames (NaN if none voiced)."""
    x = x.astype(np.float64)
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 500), int(sr / 70)
    thr = 0.3 * np.sqrt(np.mean(x ** 2))
    f0s = []
    for st in range(0, len(x) - frame, hop):
        w = x[st:st + frame] * np.hanning(frame)
        if np.sqrt(np.mean(w ** 2)) < thr:
            continue
        ac = np.correlate(w, w, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.45:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else float("nan")


def load_engine() -> IrodoriOnnx:
    return IrodoriOnnx(ROOT / "models/irodori-mf-a7542a4/v4.1-small-mf", 8)


def main() -> None:
    eng = load_engine()
    print(f"{'case':8s} " + " ".join(f"{i:>5d}" for i in range(len(SENTENCES))) + "   min  median")
    for name, cap in CASES.items():
        f0s = []
        for i, s in enumerate(SENTENCES):
            audio, _, _ = eng.synth(s, seed=0, caption=cap)
            sf.write(OUT / f"{name}_{i}.wav", audio, 48000, subtype="PCM_16")
            f0s.append(median_f0(audio))
        a = np.array(f0s)
        print(f"{name:8s} " + " ".join(f"{v:5.0f}" for v in a) + f"  {np.nanmin(a):5.0f} {np.nanmedian(a):6.0f}", flush=True)


if __name__ == "__main__":
    main()
