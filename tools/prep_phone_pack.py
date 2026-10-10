"""Build the phone pack: only the graphs the no-reference path needs, plus fixtures for on-device self-tests.

Output layout (copy the whole folder to the phone):
  fereko-pack/
    pack.json                 file list with size + sha256, pinned source revision
    text_encoder.onnx  duration.onnx  context_kv.onnx  dit_step.onnx  dacvae_decode.onnx   (hard links)
    tokenizer.json
    noref_speaker.f32         speaker_encoder output for the no-reference input (constant), float32 LE
    fixtures/tokens.json      normalized text -> token ids (from the `tokenizers` reference)
    fixtures/parity_*.f32     fixed noise + expected latent/audio for the parity check
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
import pc_synth  # noqa: E402

SRC = Path(sys.argv[1])        # .../v4.1-small-mf
GOLDEN = Path(sys.argv[2])     # golden dump dir from pc_synth --pad trim
OUT = Path(sys.argv[3])        # .../fereko-pack
GRAPHS = ["text_encoder", "duration", "context_kv", "dit_step", "dacvae_decode"]

FIXTURE_TEXTS = [
    "こんにちは。今日はいい天気ですね。", "おはよう！今日は何をしようか？", "「ねえ……ちょっと待って～」",
    "ふぇれ子だよ。よろしくね😊", "えへへ、ありがと♥", "うーん……どうしようかなぁ。",
    "明日の天気は晴れのち曇り、最高気温は23℃です。", "Hello, world! これはテストです。",
    "ＡＢＣ１２３と全角の英数字。", "半角ｶﾀｶﾅもまざってる。", "東京タワーとスカイツリー、どっちが好き？",
    "𠮷野家で牛丼を食べた。", "☃️❄️雪だるま！", "안녕하세요、韓国語もまざる。", "100円+200円=300円",
    "わたし、ずっと待ってたんだよ？", "……", "(笑)", "【お知らせ】メンテナンスは終了しました。",
    "えっ、ほんとに!?", "ふふっ💋 こっちおいで🫶", "スペース が ある 文", "改行\tとタブ",
    "〜だよね〜", "ゔぁ ゔぃ ゔ", "々〆ヶ", "ﾌｪﾚｺ", "ねえ、聞いてる？ ねえってば！！",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fixtures").mkdir(exist_ok=True)
    for g in GRAPHS:
        dst = OUT / f"{g}.onnx"
        if not dst.exists():
            os.link(SRC / f"{g}.onnx", dst)
    shutil.copyfile(SRC / "tokenizer/tokenizer.json", OUT / "tokenizer.json")

    spk = np.load(GOLDEN / "speaker_state.npy").astype("<f4")
    assert spk.shape == (1, 2, 768), spk.shape
    assert not np.load(GOLDEN / "speaker_mask.npy").any(), "no-ref speaker mask must be all zero"
    spk.tofile(OUT / "noref_speaker.f32")

    tok = Tokenizer.from_file(str(SRC / "tokenizer/tokenizer.json"))
    fixtures = []
    for t in FIXTURE_TEXTS:
        n = pc_synth.normalize_text(t).strip()
        if not n:
            continue
        ids = [1] + tok.encode(n, add_special_tokens=False).ids[:255]
        fixtures.append({"text": t, "normalized": n, "ids": ids})
    (OUT / "fixtures/tokens.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = json.loads((GOLDEN / "meta.json").read_text(encoding="utf-8"))
    for name in ["noise", "latent", "audio"]:
        np.load(GOLDEN / f"{name}.npy").astype("<f4").tofile(OUT / f"fixtures/parity_{name}.f32")
    (OUT / "fixtures/parity.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "pack.json":
            rel = p.relative_to(OUT).as_posix()
            print("hash", rel, flush=True)
            files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256(p)})
    pack = {
        "schema": 1,
        "source": "raratu/Onsei-Irodori-v4.1-Small-MF@a7542a4f1ea0fd8d44ad82d7c0fd7bb8b81ed641 (ios-mf-v2)",
        "reference_impl": "Aratako/Irodori-TTS@89f9d8fbd4d51ea019867ee1197725ede1df13c5",
        "bos_id": 1, "pad_id": 3, "sample_rate": 48000, "hop": 1920, "latent_dim": 32,
        "num_layers": 12, "num_steps": 4, "text_max": 256, "speaker_len": 2,
        "files": files,
    }
    (OUT / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f["size"] for f in files)
    print(f"pack: {len(files)} files, {total / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
