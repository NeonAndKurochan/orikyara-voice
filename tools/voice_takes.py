"""Candidate takes for ふぇれ子's reference voice: one in-character line, caption from chat/voice.txt, several seeds.

Writes out/voice/take_<seed>.wav and the latent of each take (take_<seed>.latent.npy). The latent of the
chosen take becomes the reference (speaker) input, so no re-encoding of the audio is needed.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from caption_eval import load_engine, median_f0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/voice"
OUT.mkdir(parents=True, exist_ok=True)
CAPTION = (ROOT / "chat/voice.txt").read_text(encoding="utf-8").strip()
LINE = "…別に、キミのこと待ってたわけじゃないけど。まあ、来たならちょっとくらい話してあげてもいいよ。"
SEEDS = [1, 2, 3, 4, 5, 6]

eng = load_engine()
for seed in SEEDS:
    audio, rep, ten = eng.synth(LINE, seed=seed, caption=CAPTION)
    sf.write(OUT / f"take_{seed}.wav", audio, 48000, subtype="PCM_16")
    np.save(OUT / f"take_{seed}.latent.npy", ten["latent"][:, : rep["flatten_frame"] or rep["latent_frames"]])
    print(f"seed {seed}: {rep['audio_sec']:.1f}s  pitch {median_f0(audio):.0f} Hz", flush=True)
