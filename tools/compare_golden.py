"""Compare two golden dumps from pc_synth.py (e.g. full vs trimmed padding)."""
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])


def load(d: Path, n: str) -> np.ndarray:
    return np.load(d / f"{n}.npy")


n = int(min(load(a, "text_mask").sum(), load(b, "text_mask").sum()))
print("text_state[:n] maxdiff", float(np.abs(load(a, "text_state")[:, :n] - load(b, "text_state")[:, :n]).max()))
print("log_frames", load(a, "log_frames"), load(b, "log_frames"))
for k in ["v0", "v3", "latent", "audio"]:
    x, y = load(a, k), load(b, k)
    if x.shape != y.shape:
        m = min(x.shape[-1], y.shape[-1])
        print(k, "shape", x.shape, y.shape)
        x, y = x[..., :m], y[..., :m]
    d = np.abs(x - y).max()
    print(f"{k:7s} maxdiff {d:.3e}  rel {d / np.abs(x).max():.3e}")
