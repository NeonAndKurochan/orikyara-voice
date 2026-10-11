"""See-through layers -> picture parts for the 2D mode (web/flat.mjs): "visible = original picture, hidden = See-through".

  <ComfyUI venv python> -I flat_bake.py <See-through *_layers.json> <source picture> <out dir>
        [--cutout <RGBA cut-out of the same picture>] [--split-dark <layer>=<dark name>,<light name>] [--size 1280]

See-through (ComfyUI tk_seethrough, SavePSD node) writes one PNG per layer and *_layers.json (name, filename,
left, top) on a square canvas: the source picture padded to a square and resized to --size. (Its own
original.png is stretched to the square instead, so it does not line up with the layers; the source picture
is padded here the same way the layers were.)

1. Every figure pixel gets one owner: of the layers covering it, the one whose colour is closest to the source
   picture there, so where the arm layer's filled-in hidden part overlaps the chest, the chest owns it. Colours
   within MARGIN of the best are a tie (a white sleeve over a white coat): the layer See-through put on top wins,
   and tied pixels are never cut away.
2. The draw order comes from who owns the overlaps (pairwise votes, greedy ordering), not from See-through.
3. Each layer: owned pixels take the source picture's colour (solid inside, the layers' own soft alpha on the
   outline), so the still picture is the source picture with no seams; filled-in parts under a layer drawn
   above are kept (they show when parts move). Pixels clearly owned by a layer drawn below (the coat hanging
   over the sleeve) are cut away and become "<name>__back", drawn at the very bottom: hidden while still,
   and what fills the hole when the part moves.
4. Transparent pixels get the nearest opaque colour, so texture filtering leaves no fringes.

--cutout: figure pixels no layer covers (e.g. animal ears See-through dropped) become a layer "extra".
--split-dark: one layer split by brightness into two (e.g. topwear=sweater,coat), so a part can sit between.
Then write rig.json by hand (joint positions in canvas pixels, see web/flat.mjs) next to layers_flat.json.
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('layers_json')
ap.add_argument('source')
ap.add_argument('out')
ap.add_argument('--cutout')
ap.add_argument('--split-dark')
ap.add_argument('--size', type=int, default=1280)
args = ap.parse_args()
N = args.size
MARGIN = 18
os.makedirs(args.out, exist_ok=True)


def padded(path, mode):
    """the picture as the layers see it: centred on a square, resized to N"""
    im = Image.open(path).convert(mode)
    S = max(im.size)
    sq = Image.new(mode, (S, S), (255, 255, 255) if mode == 'RGB' else (0, 0, 0, 0))
    sq.paste(im, ((S - im.width) // 2, (S - im.height) // 2))
    return np.asarray(sq.resize((N, N), Image.LANCZOS)).astype(np.float32)


orig = padded(args.source, 'RGB')
info = json.load(open(args.layers_json, encoding='utf-8'))
src_dir = os.path.dirname(args.layers_json)
split = None
if args.split_dark:
    k, v = args.split_dark.split('=')
    split = (k, *v.split(','))

names, rgba = [], []
for l in info['layers'] if isinstance(info, dict) else info:
    im = np.asarray(Image.open(os.path.join(src_dir, l['filename'])).convert('RGBA')).astype(np.float32)
    c = np.zeros((N, N, 4), np.float32)
    c[l['top']:l['top'] + im.shape[0], l['left']:l['left'] + im.shape[1]] = im
    if split and l['name'] == split[0]:
        a = c[:, :, 3] > 0
        luma = c[:, :, :3] @ np.array([0.299, 0.587, 0.114], np.float32)
        dark = ndimage.binary_opening(ndimage.binary_closing(a & (luma < 110), iterations=3), iterations=2)
        for nm, m in ((split[1], dark), (split[2], a & ~dark)):
            p = c.copy()
            p[~m, 3] = 0
            names.append(nm)
            rgba.append(p)
        continue
    names.append(l['name'])
    rgba.append(c)

if args.cutout:
    cut = padded(args.cutout, 'RGBA')
    covered = np.stack([r[:, :, 3] > 100 for r in rgba]).any(0)
    rest = (cut[:, :, 3] > 100) & ~covered
    lab, n = ndimage.label(rest)
    sizes = ndimage.sum(rest, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s > 150])
    keep = ndimage.binary_dilation(keep, iterations=2) & (cut[:, :, 3] > 0)
    if keep.any():
        e = cut.copy()
        e[~keep, 3] = 0
        names.append('extra')
        rgba.append(e)
        print('extra (from the cut-out):', int(keep.sum()), 'px')

L = len(names)
alpha = np.stack([r[:, :, 3] / 255.0 for r in rgba])
fig_a = alpha.max(0)          # the figure's outline: the union of the layers
fig = fig_a > 0.02
cover = alpha > 0.5

# 1. owners
blur = cv2.GaussianBlur(orig, (5, 5), 0)
dist = np.full((L, N, N), np.inf, np.float32)
for i, r in enumerate(rgba):
    d = np.sqrt(((cv2.GaussianBlur(r[:, :, :3], (5, 5), 0) - blur) ** 2).sum(2))
    dist[i][cover[i]] = d[cover[i]]
best = dist.min(0)
tie = dist <= best[None] + MARGIN
owner = (L - 1) - np.argmax(tie[::-1], 0)    # the highest See-through index among the tied
clear = ~tie                                 # layer k is clearly worse than the owner here
gap = fig & ~cover.any(0)
if gap.any():                                # thin gaps and outline pixels: the nearest covered layer
    _, (iy, ix) = ndimage.distance_transform_edt(~cover.any(0), return_indices=True)
    owner[gap] = owner[iy[gap], ix[gap]]
owner[~fig] = -1

# 2. draw order
above = np.zeros((L, L))
for i in range(L):
    for j in range(i + 1, L):
        ov = cover[i] & cover[j] & fig & (clear[i] | clear[j])
        oi, oj = int((owner[ov] == i).sum()), int((owner[ov] == j).sum())
        if oi + oj < 20:
            continue
        if oi > oj:
            above[i, j] = oi - oj
        else:
            above[j, i] = oj - oi
remaining, order = list(range(L)), []        # bottom to top
while remaining:
    k = min(remaining, key=lambda k: (sum(above[k, m] for m in remaining) - sum(above[m, k] for m in remaining), k))
    order.append(k)
    remaining.remove(k)
rank = np.empty(L, int)
rank[order] = np.arange(L)
print('order:', ' < '.join(names[k] for k in order))

# 3. bake
meta, backs = [], []


def save(name, rgb, a):
    m = a > 0.01
    if m.sum() < 20:
        return None
    _, (iy, ix) = ndimage.distance_transform_edt(~m, return_indices=True)
    rgb = rgb[iy, ix]
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(0, ys.min() - 2), min(N, ys.max() + 3), max(0, xs.min() - 2), min(N, xs.max() + 3)
    fn = name.replace(' ', '_') + '.png'
    Image.fromarray(np.dstack([rgb, a * 255]).clip(0, 255).astype(np.uint8)[y0:y1, x0:x1]).save(os.path.join(args.out, fn))
    return {'name': name, 'filename': fn, 'left': int(x0), 'top': int(y0)}


for k in order:
    a = alpha[k].copy()
    own = owner == k
    cut_away = (owner >= 0) & (owner != k) & (rank[np.clip(owner, 0, L - 1)] < rank[k]) & clear[k] & (a > 0.01)
    if cut_away.sum() >= 200:
        b = save(names[k] + '__back', rgba[k][:, :, :3].copy(), np.where(cut_away, alpha[k], 0))
        if b:
            backs.append(b)
    a[cut_away] = 0
    a[~fig] = 0
    a[own] = fig_a[own]
    inner = own & (fig_a > 0.98)
    a[inner] = 1
    rgb = rgba[k][:, :, :3].copy()
    rgb[inner] = orig[inner]
    m = save(names[k], rgb, a)
    if m:
        meta.append(m)
    else:
        print('  (empty after baking:', names[k] + ')')
if backs:
    print('back pieces:', ', '.join(b['name'] for b in backs))
meta = backs + meta
json.dump(meta, open(os.path.join(args.out, 'layers_flat.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# check: the still picture against the source
comp = np.zeros((N, N, 3), np.float32)
for l in meta:
    im = np.asarray(Image.open(os.path.join(args.out, l['filename']))).astype(np.float32) / 255
    h, w = im.shape[:2]
    reg = comp[l['top']:l['top'] + h, l['left']:l['left'] + w]
    reg[:] = im[:, :, :3] * im[:, :, 3:] + reg * (1 - im[:, :, 3:])
inside = fig_a > 0.98
print(len(meta), 'layers; still picture vs source, mean abs diff inside the figure:',
      round(float(np.abs(comp[inside] * 255 - orig[inside]).mean()), 3))
