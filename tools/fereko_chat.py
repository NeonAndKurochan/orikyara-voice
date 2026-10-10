r"""ふぇれ子と文字で話す（PC版）。返事は Claude（Cloudflare の中継経由）、声は ONNX 版 Irodori（端末内）。

  .venv\Scripts\python.exe -X utf8 tools\fereko_chat.py [--no-voice]

API キーは中継（Cloudflare Worker のシークレット）にだけ置く。この PC には中継の合言葉
（%USERPROFILE%\.fereko\relay-token.txt）しかない。モデル・effort・1日の上限も中継が決める。
会話は chat/history.json に追記していく。
コマンド: /quit 終了  /voice 声のオンオフ  /cost 今日の使用額  /reset 会話を最初から（履歴は別名で残す）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import threading
import time
import winsound
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relay_client import Relay, RelayError  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHAT = ROOT / "chat"
HISTORY = CHAT / "history.json"
PERSONA = CHAT / "persona.md"
VOICE_CAPTION = CHAT / "voice.txt"   # Irodori の声の説明文（キャプション）。空なら説明なし
VOICE_REF = CHAT / "voice_ref.latent.npy"   # 登録した声（選んだ候補の潜在表現）。なければ参考音声なし
PACK = ROOT / "models/irodori-mf-a7542a4/v4.1-small-mf"

# A sentence = text up to its end punctuation, plus any emoji/spaces right after it
# (the persona puts a style emoji at the end of a sentence; it belongs to that sentence).
SENTENCE = re.compile(
    r"[^。！？!?…]+[。！？!?…]*[\s\U0001F300-\U0001FAFF☀-➿️‍]*"
    r"|[。！？!?…]+[\s\U0001F300-\U0001FAFF☀-➿️‍]*"
)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE.findall(text)]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        # glue very short fragments (e.g. a lone emoji) to the previous sentence
        if out and len(p) <= 2:
            out[-1] += p
        else:
            out.append(p)
    return out


class Voice:
    """Speaks sentences in order: synthesizes the next one while the current one plays."""

    def __init__(self, threads: int = 8) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        from pc_synth import IrodoriOnnx  # heavy import, only when voice is on

        t0 = time.perf_counter()
        self.engine = IrodoriOnnx(PACK, threads)
        self.tmp = Path(tempfile.mkdtemp(prefix="fereko-voice-"))
        self.lock = threading.Lock()
        # Without a caption the voice is chosen at random per sentence and is sometimes male;
        # a caption keeps it in range (tools/caption_eval2.py: 0 of 24 low-pitched vs 1 of 24).
        self.caption = VOICE_CAPTION.read_text(encoding="utf-8").strip() if VOICE_CAPTION.exists() else ""
        # The registered voice (latent of the take the user picked): keeps the same voice across sentences.
        import numpy as np
        self.ref = np.load(VOICE_REF) if VOICE_REF.exists() else None
        print(f"（声の準備 {time.perf_counter() - t0:.1f} 秒）")

    def speak(self, text: str, seed: int = 0) -> None:
        with self.lock:
            play_until = 0.0
            for i, sentence in enumerate(split_sentences(text)):
                try:
                    audio, report, _ = self.engine.synth(sentence, seed=seed, caption=self.caption, ref_latent=self.ref)
                except ValueError:
                    continue  # nothing speakable after normalization
                wav = self.tmp / f"s{i % 2}.wav"
                wait = play_until - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                sf.write(wav, audio, 48000, subtype="PCM_16")
                winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
                play_until = time.perf_counter() + report["audio_sec"] + 0.15
            wait = play_until - time.perf_counter()
            if wait > 0:
                time.sleep(wait)


class Fereko:
    def __init__(self) -> None:
        self.relay = Relay()
        self.system = PERSONA.read_text(encoding="utf-8")
        self.messages: list[dict] = load_json(HISTORY, [])

    def usage_line(self) -> str:
        try:
            u = self.relay.usage()
            return f"今日 ${u['usd']:.4f} / 上限 ${u['limit_usd']:.2f}（{u['requests']} 回、Opus 5.5 の定価で計算した目安）"
        except (RelayError, OSError) as e:
            return f"（使用額を取れなかった: {e}）"

    def reply(self, user_text: str, on_text) -> str | None:
        request_messages = self.messages + [{"role": "user", "content": user_text}]
        try:
            r = self.relay.chat(self.system, request_messages, on_text)
        except RelayError as e:
            hint = {401: "中継の合言葉が合わないみたい", 429: "今日の上限に達したみたい", 503: "中継に API キーがまだ入ってないみたい"}
            print(f"\n（{hint.get(e.status, '中継でエラー')}: {e}）")
            return None
        except OSError as e:
            print(f"\n（中継につながらないみたい: {e}）")
            return None

        if r.stop_reason == "refusal":
            print("\n（この話題は答えられないって返ってきた。履歴には入れないね）")
            return None
        # Append-only history: keep the assistant content blocks exactly as streamed
        # (thinking blocks must be sent back unchanged on later turns).
        self.messages = request_messages + [{"role": "assistant", "content": r.content}]
        save_json(HISTORY, self.messages)
        u = r.usage
        note = "" if r.model in (None, "claude-opus-5-5") else f"  {r.model} が答えた"
        print(f"\n   （入力 {u.get('input_tokens', 0)}+キャッシュ {u.get('cache_read_input_tokens', 0)}  出力 {u.get('output_tokens', 0)}{note}）")
        return r.text

    def reset(self) -> None:
        if HISTORY.exists():
            HISTORY.rename(CHAT / f"history-{datetime.now():%Y%m%d-%H%M%S}.json")
        self.messages = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    CHAT.mkdir(exist_ok=True)
    fereko = Fereko()
    voice = None if args.no_voice else Voice(args.threads)
    print(f"ふぇれ子とおしゃべり（{fereko.usage_line()}）")
    print("/quit で終了、/voice で声のオンオフ、/cost で使用額、/reset で会話をやり直し\n")

    while True:
        try:
            line = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/cost":
            print(fereko.usage_line())
            continue
        if line == "/reset":
            fereko.reset()
            print("（会話を最初からにした。前の履歴は chat/ に別名で残してある）")
            continue
        if line == "/voice":
            if voice is None:
                voice = Voice(args.threads)
            else:
                voice = None
            print("（声 " + ("オン" if voice else "オフ") + "）")
            continue

        print("ふぇれ子> ", end="", flush=True)
        text = fereko.reply(line, lambda t: print(t, end="", flush=True))
        if text and voice:
            voice.speak(text)


if __name__ == "__main__":
    main()
