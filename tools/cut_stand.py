"""Cuts ふぇれ子's stand picture out of its plain light background (for the new chat screen).

  <ComfyUI venv python> -I cut_stand.py <in.webp> <out.png> [<preview.png>]

Flood fill from the image border over pixels close to the background colour, so white hair
and the white coat inside the outline are kept. The edge is softened by a small blur of the mask.
"""
import sys

import cv2
import numpy as np
from PIL import Image

src = Image.open(sys.argv[1]).convert("RGB")
a = np.asarray(src).astype(np.int16)
h, w, _ = a.shape
bg = np.median(np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)]), axis=0)
diff = np.abs(a - bg).max(axis=2)
cand = (diff <= 14).astype(np.uint8)          # looks like background

# keep only background-like pixels connected to the border
n, labels = cv2.connectedComponents(cand, connectivity=4)
border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
bgmask = np.isin(labels, list(border)) & (cand == 1)
# (enclosed pockets are kept: highlights in the hair and coat are exactly the background colour)

fg = (~bgmask).astype(np.uint8)
# drop small islands (background texture that differed a little), keep the figure
n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 1500]
fg = np.isin(labels, keep).astype(np.uint8)
alpha = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32)
alpha = cv2.GaussianBlur(alpha, (3, 3), 0.8)
out = np.dstack([np.asarray(src), (alpha * 255).astype(np.uint8)])
Image.fromarray(out, "RGBA").save(sys.argv[2])
print("bg", bg, "kept", round(float(alpha.mean()), 3))

if len(sys.argv) > 3:   # over a dark and a green background, to spot holes or halos
    rgba = Image.fromarray(out, "RGBA")
    prev = Image.new("RGB", (w * 2, h), (30, 30, 36))
    prev.paste(Image.new("RGB", (w, h), (40, 160, 70)), (w, 0))
    prev.paste(rgba, (0, 0), rgba)
    prev.paste(rgba, (w, 0), rgba)
    prev.resize((w, h // 2)).save(sys.argv[3])
