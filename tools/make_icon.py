"""App icon (Android adaptive icon) from the cut-out stand picture: ふぇれ子's head and shoulders
over a dark charcoal background with a faint red glow (her eye colour).

  <ComfyUI venv python> -I make_icon.py <fereko_stand_cut.png> <res dir> [<preview.png>]

Writes mipmap-xxxhdpi/ic_launcher_fg.png and ic_launcher_bg.png (432 px = 108 dp at 4x; the
launcher masks it to its own shape, showing about the middle 72 dp) and a legacy 192 px icon.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 432
src = Image.open(sys.argv[1]).convert("RGBA")
res = Path(sys.argv[2])

# background: radial charcoal, a soft red glow behind the head
yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
r = np.hypot(xx - SIZE / 2, yy - SIZE * 0.42) / (SIZE / 2)
base = np.clip(1 - r * 0.55, 0, 1)
bg = np.stack([28 + 22 * base, 28 + 20 * base, 34 + 22 * base], axis=-1)
glow = np.exp(-(r / 0.55) ** 2)[..., None] * np.array([70, 18, 22])
bg_img = Image.fromarray(np.clip(bg + glow, 0, 255).astype(np.uint8), "RGB")

# foreground: head and shoulders, the body running off the bottom edge
crop = src.crop((285, 0, 795, 510))            # ears to shoulders, centred on the face
fg = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
head = crop.resize((352, 352), Image.LANCZOS)
fg.alpha_composite(head, ((SIZE - 352) // 2, SIZE - 352 + 6))
# a thin dark outline so white hair stays readable on light launcher wallpapers
a = fg.getchannel("A").filter(ImageFilter.MaxFilter(5))
outline = Image.new("RGBA", (SIZE, SIZE), (16, 16, 20, 0))
outline.putalpha(a.point(lambda v: int(v * 0.55)))
fg = Image.alpha_composite(outline, fg)

out = res / "mipmap-xxxhdpi"
out.mkdir(parents=True, exist_ok=True)
fg.save(out / "ic_launcher_fg.png")
bg_img.save(out / "ic_launcher_bg.png")

# legacy square icon (older launchers / settings lists): the middle 72 dp, rounded
full = Image.alpha_composite(bg_img.convert("RGBA"), fg)
inner = full.crop((SIZE // 6, SIZE // 6, SIZE * 5 // 6, SIZE * 5 // 6)).resize((192, 192), Image.LANCZOS)
mask = Image.new("L", (192, 192), 0)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, 191, 191), 44, fill=255)
legacy = Image.new("RGBA", (192, 192), (0, 0, 0, 0))
legacy.paste(inner, (0, 0), mask)
legacy.save(out / "ic_launcher.png")

if len(sys.argv) > 3:   # how launchers will show it: circle and rounded square, on light and dark
    tiles = []
    for shape in ("circle", "squircle"):
        m = Image.new("L", (SIZE, SIZE), 0)
        d = ImageDraw.Draw(m)
        box = (SIZE // 6, SIZE // 6, SIZE * 5 // 6, SIZE * 5 // 6)
        if shape == "circle":
            d.ellipse(box, fill=255)
        else:
            d.rounded_rectangle(box, 70, fill=255)
        t = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        t.paste(full, (0, 0), m)
        tiles.append(t.crop(box))
    w = tiles[0].width
    prev = Image.new("RGB", (w * 4 + 50, w + 20), (235, 235, 240))
    dark = Image.new("RGB", (w * 2 + 30, w + 20), (20, 20, 24))
    prev.paste(dark, (w * 2 + 20, 0))
    for i, t in enumerate(tiles * 2):
        prev.paste(t, (10 + i * (w + 10), 10), t)
    prev.save(sys.argv[3])
print("ok")
