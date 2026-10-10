"""Irodori v4.1-Small-MF (raratu ONNX pack, ios-mf-v2) end-to-end on PC with ONNX Runtime CPU.

Reference pipeline for the Android probe: same graphs, same pre/post-processing.
Follows Aratako/Irodori-TTS @ 89f9d8f (inference_runtime.synthesize, meanflow.sample_euler_meanflow).
No reference audio, empty caption.

Usage:
  python -I pc_synth.py --pack <dir of v4.1-small-mf> --text "..." --out out.wav
         [--pad full|trim] [--seed 0] [--threads 8] [--dump golden_dir]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tokenizers import Tokenizer

SAMPLE_RATE = 48000
HOP = 1920
LATENT_DIM = 32
NUM_LAYERS = 12
NUM_STEPS = 4
TEXT_MAX = 256
CAPTION_MAX = 512
MIN_SECONDS, MAX_SECONDS = 0.5, 30.0
SPEAKER_PATCH = 4

# ---- text normalization (copy of irodori_tts/text_normalization.py @ 89f9d8f, MIT) ----
SIMPLE_REPLACE_MAP = {
    "\t": "", "[n]": "", r"\[n\]": "", "　": "", "？": "?", "！": "!",
    "♥": "♡", "●": "○", "◯": "○", "〇": "○",
}
REGEX_REPLACE_MAP = {
    re.compile(r"[;▼♀♂《》≪≫①②③④⑤⑥]"): "",
    re.compile(r"[˗‐-―⁃−⎯⏤─━⸺⸻]"): "",
    re.compile(r"[～〜]"): "ー",
    re.compile(r"…{3,}"): "……",
}


def strip_outer_brackets(text: str) -> str:
    pairs = {"「": "」", "『": "』", "（": "）", "【": "】", "(": ")"}
    while len(text) >= 2:
        s, e = text[0], text[-1]
        if s in pairs and pairs[s] == e:
            depth, enclosing = 0, True
            for i, ch in enumerate(text):
                if ch == s:
                    depth += 1
                elif ch == e:
                    depth -= 1
                if depth == 0 and i < len(text) - 1:
                    enclosing = False
                    break
            if enclosing and depth == 0:
                text = text[1:-1]
                continue
        break
    return text


def normalize_text(text: str) -> str:
    for old, new in SIMPLE_REPLACE_MAP.items():
        text = text.replace(old, new)
    for pat, rep in REGEX_REPLACE_MAP.items():
        text = pat.sub(rep, text)
    text = strip_outer_brackets(text)
    text = unicodedata.normalize("NFKC", text)
    return text.replace("...", "…").replace("..", "…")


# ---- tokenizer: explicit BOS, no automatic special tokens, right padding ----
def encode(tok: Tokenizer, text: str, max_len: int, pad_to: int | None, bos: int, pad: int):
    body = tok.encode(text, add_special_tokens=False).ids[: max_len - 1]
    ids = [bos] + body
    length = pad_to if pad_to is not None else len(ids)
    mask = [1.0] * len(ids) + [0.0] * (length - len(ids))
    ids = ids + [pad] * (length - len(ids))
    return np.array([ids], dtype=np.int64), np.array([mask], dtype=np.float32)


def find_flattening_point(latent: np.ndarray, window=20, std_thr=0.05, mean_thr=0.1) -> int:
    total = latent.shape[0]
    padded = np.concatenate([latent, np.zeros((window, latent.shape[1]), latent.dtype)], 0)
    for i in range(padded.shape[0] - window):
        w = padded[i : i + window]
        if w.std() < std_thr and abs(w.mean()) < mean_thr:
            return i
    return total


class IrodoriOnnx:
    """Loads the ONNX graphs once; synth() can then be called many times."""

    GRAPHS = ["text_encoder", "speaker_encoder", "duration", "context_kv", "dit_step", "dacvae_decode"]

    def __init__(self, pack: Path, threads: int = 8, files: dict[str, Path] | None = None) -> None:
        """files: optional per-graph override, e.g. {"dit_step": Path(".../dit_step.nbits4.onnx")}."""
        pack = Path(pack)
        files = files or {}
        tcfg = json.loads((pack / "tokenizer/tokenizer_config.json").read_text(encoding="utf-8"))
        self.tok = Tokenizer.from_file(str(pack / "tokenizer/tokenizer.json"))
        self.bos = self.tok.token_to_id(tcfg["bos_token"])
        self.pad = self.tok.token_to_id(tcfg["pad_token"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.load_ms: dict[str, float] = {}
        self.sess: dict[str, ort.InferenceSession] = {}
        for name in self.GRAPHS:
            t0 = time.perf_counter()
            path = files.get(name, pack / f"{name}.onnx")
            self.sess[name] = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.load_ms[name] = (time.perf_counter() - t0) * 1000
        self.kv_names = [o.name for o in self.sess["context_kv"].get_outputs()]

    def synth(self, raw_text: str, seed: int = 0, pad_mode: str = "trim", noise: np.ndarray | None = None,
              caption: str = "", ref_latent: np.ndarray | None = None):
        """Returns (audio float32 mono 48 kHz, report dict, tensors dict).

        ref_latent: optional reference voice as a DAC-VAE latent (1, T, 32), e.g. the latent of a
        chosen take. It goes through the speaker encoder; without it the voice is random per call.
        """
        timings: dict[str, float] = {}

        def run(name, feeds):
            t0 = time.perf_counter()
            out = self.sess[name].run(None, feeds)
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
            return out

        t_start = time.perf_counter()
        text = normalize_text(raw_text).strip()
        if not text:
            raise ValueError("text became empty after normalization")
        full = pad_mode == "full"
        text_ids, text_mask = encode(self.tok, text, TEXT_MAX, TEXT_MAX if full else None, self.bos, self.pad)
        # Caption (voice description) is tokenized like the text, without normalize_text (upstream strips it only).
        caption = caption.strip()
        cap_ids, cap_mask = encode(self.tok, caption, CAPTION_MAX, CAPTION_MAX if full else None, self.bos, self.pad)
        if not caption:
            cap_mask[:] = 0.0  # empty caption: upstream zeroes the whole mask

        text_state, _ = run("text_encoder", {"input_ids": text_ids, "mask": text_mask})
        _, caption_state = run("text_encoder", {"input_ids": cap_ids, "mask": cap_mask})

        if ref_latent is None:
            # no reference: zeros of speaker_patch_size frames, all masked (inference_runtime._load_reference_latent)
            ref = np.zeros((1, SPEAKER_PATCH, LATENT_DIM), np.float32)
            ref_mask = np.zeros((1, SPEAKER_PATCH), np.float32)
        else:
            t_ref = ref_latent.shape[1] // SPEAKER_PATCH * SPEAKER_PATCH   # whole speaker patches only
            ref = np.ascontiguousarray(ref_latent[:, :t_ref], np.float32)
            ref_mask = np.ones((1, t_ref), np.float32)
        spk_state, spk_mask = run("speaker_encoder", {"ref_latent": ref, "mask": ref_mask})
        has_speaker = 1.0 if ref_latent is not None else 0.0

        (log_frames,) = run("duration", {
            "text_state": text_state, "text_mask": text_mask,
            "speaker_state": spk_state, "has_speaker": np.full((1,), has_speaker, np.float32),
            "caption_state": caption_state, "caption_mask": cap_mask,
            "has_caption": np.full((1,), 1.0 if caption else 0.0, np.float32),
        })
        pred_frames = float(np.expm1(log_frames).mean())
        min_frames = max(1, math.ceil(MIN_SECONDS * SAMPLE_RATE / HOP))
        max_frames = max(1, math.floor(MAX_SECONDS * SAMPLE_RATE / HOP))
        steps_t = max(min_frames, min(max_frames, int(round(pred_frames))))
        target_samples = steps_t * HOP

        kv = run("context_kv", {"text_state": text_state, "speaker_state": spk_state, "caption_state": caption_state})
        kv_feeds = dict(zip(self.kv_names, kv))

        if noise is not None:
            x = noise.astype(np.float32)
            if x.shape[1] != steps_t:
                raise ValueError(f"noise length {x.shape[1]} != predicted frames {steps_t}")
        else:
            x = np.random.default_rng(seed).standard_normal((1, steps_t, LATENT_DIM)).astype(np.float32)
        noise0 = x.copy()
        schedule = np.linspace(1.0, 0.0, NUM_STEPS + 1, dtype=np.float32)
        velocities = []
        for i in range(NUM_STEPS):
            t, nxt = schedule[i], schedule[i + 1]
            (v,) = run("dit_step", {
                "x_t": x, "t": np.array([t], np.float32), "delta_t": np.array([t - nxt], np.float32),
                "text_mask": text_mask, "speaker_mask": spk_mask, "caption_mask": cap_mask,
                "speaker_kv_scales": np.ones((NUM_LAYERS,), np.float32), **kv_feeds,
            })
            velocities.append(v)
            x = x + v * (nxt - t)

        (wave,) = run("dacvae_decode", {"latent": x})
        flat = find_flattening_point(x[0])
        max_samples = target_samples
        if flat * HOP > 0:
            max_samples = min(max_samples, flat * HOP)
        audio = wave[0, 0, :max_samples]
        synth_sec = time.perf_counter() - t_start
        audio_sec = len(audio) / SAMPLE_RATE
        report = {
            "text": raw_text, "normalized_text": text, "token_ids": text_ids[0, : int(text_mask.sum())].tolist(),
            "pad": pad_mode, "log_frames": float(log_frames[0]), "pred_frames": round(pred_frames, 2),
            "latent_frames": steps_t, "flatten_frame": flat, "audio_samples": len(audio),
            "audio_sec": round(audio_sec, 3), "synth_sec": round(synth_sec, 3),
            "rtf": round(synth_sec / max(audio_sec, 1e-6), 3), "peak": float(np.abs(audio).max()),
            "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
        }
        tensors = {
            "text_ids": text_ids, "text_mask": text_mask, "text_state": text_state,
            "caption_state": caption_state, "speaker_state": spk_state, "speaker_mask": spk_mask,
            "log_frames": log_frames, "noise": noise0, "v0": velocities[0], "v3": velocities[3],
            "latent": x, "audio": audio,
        }
        return audio, report, tensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", choices=["full", "trim"], default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dump", default=None, help="directory for golden .npy tensors")
    ap.add_argument("--noise", default=None, help="load initial noise .npy instead of seeding")
    args = ap.parse_args()

    eng = IrodoriOnnx(Path(args.pack), args.threads)
    noise = np.load(args.noise) if args.noise else None
    audio, report, tensors = eng.synth(args.text, args.seed, args.pad, noise)
    report["timings_ms"].update({f"load_{k}": round(v, 1) for k, v in eng.load_ms.items()})
    sf.write(args.out, audio, SAMPLE_RATE, subtype="PCM_16")
    print(json.dumps(report, ensure_ascii=False, indent=1))

    if args.dump:
        d = Path(args.dump)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        for name, arr in tensors.items():
            np.save(d / f"{name}.npy", arr)


if __name__ == "__main__":
    main()
