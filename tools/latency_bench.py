"""Time to first text through the real relay, per model, with a cache warm-up first.

  python -I latency_bench.py <history.json> [opus sonnet haiku]

Uses chat/persona.md and the given history (e.g. pulled from the phone). Nothing is saved:
the test turns are only appended to a local copy.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relay_client import Relay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LINES = ["ねえ、今日の晩ごはん何がいいと思う？", "じゃあカレーにする。辛口と甘口どっち派？"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    persona = (ROOT / "chat/persona.md").read_text(encoding="utf-8")
    history = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    models = sys.argv[2:] or ["opus", "sonnet", "haiku"]
    relay = Relay()
    for model in models:
        msgs = list(history)
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        t = time.perf_counter()
        u = relay.warm(persona, msgs, model)
        print(f"[{model}] warm {1000 * (time.perf_counter() - t):.0f} ms  write {u.get('cache_creation_input_tokens')} read {u.get('cache_read_input_tokens')}")
        convo = list(history)
        for line in LINES:
            convo.append({"role": "user", "content": line})
            first = [0.0]
            t = time.perf_counter()

            def on_text(_d):
                if not first[0]:
                    first[0] = time.perf_counter()

            r = relay.chat(persona, convo, on_text=on_text, model=model)
            total = time.perf_counter() - t
            u = r.usage
            print(f"  first text {1000 * (first[0] - t):.0f} ms, total {1000 * total:.0f} ms, out {u.get('output_tokens')} "
                  f"read {u.get('cache_read_input_tokens')} write {u.get('cache_creation_input_tokens')} in {u.get('input_tokens')} "
                  f"model {r.model}: {r.text}")
            convo.append({"role": "assistant", "content": r.content})
    print("usage today:", relay.usage())


main()
