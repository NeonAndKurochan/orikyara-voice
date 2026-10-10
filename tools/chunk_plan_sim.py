"""Pick a decode chunk plan: simulate when playback can start without running dry.

Phone numbers (v0.2.0, 6 threads): decoder ~31.5 ms per window frame, text+DiT ~1.0 s for a
92-frame sentence (DiT scales with length). Audio is 40 ms per frame.
"""
R = 0.0315      # s per decoded window frame
FRAME = 0.04    # s of audio per frame
MARGIN = 0.15


def plan(total, sizes, ctx):
    out, st, i = [], 0, 0
    while st < total:
        en = min(total, st + sizes[min(i, len(sizes) - 1)])
        out.append((st, en, max(0, st - ctx), min(total, en + ctx)))
        st, i = en, i + 1
    return out


def start_time(total, sizes, ctx, dit_end):
    t, need = dit_end, 0.0
    for st, en, a, b in plan(total, sizes, ctx):
        t += (b - a) * R
        need = max(need, t - st * FRAME)          # chunk must be ready before its first sample is due
    first_ready = dit_end + (plan(total, sizes, ctx)[0][3]) * R
    return max(need, first_ready) + MARGIN, t


def dit_end(total):
    return 0.1 + 0.9 * total / 92


plans = {
    "whole (v0.1)": [10**6],
    "8/24/64 (v0.2)": [8, 24, 64],
    "8/16/32/64": [8, 16, 32, 64],
    "6/12/24/48/96": [6, 12, 24, 48, 96],
    "4/8/16/32/64/128": [4, 8, 16, 32, 64, 128],
    "8/16/24/40/64/96": [8, 16, 24, 40, 64, 96],
}
for ctx in (8, 6):
    print(f"--- context {ctx} frames each side")
    for name, sizes in plans.items():
        row = []
        for total in (25, 50, 92, 150, 250):
            s, fin = start_time(total, sizes, ctx if name != "whole (v0.1)" else 0, dit_end(total))
            row.append(f"{total * FRAME:4.1f}s音声→{s:4.2f}s")
        print(f"{name:18s} " + "  ".join(row))
