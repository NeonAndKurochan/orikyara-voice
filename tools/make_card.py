"""GitHub social preview card (1280x640): the character render on the right, name and pitch on the left.

  <ComfyUI venv python> -I make_card.py <render.png|jpg> <out.png> [top bottom]  (crop, as fractions of the render height)

The render is a screenshot of web/avatar.html on a flat dark background (avatarStep in a PC browser);
the card uses that background colour so the render blends in.
"""
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 640
render = Image.open(sys.argv[1]).convert("RGB")
bg = render.getpixel((8, 8))

card = Image.new("RGB", (W, H), bg)
# a faint red glow behind the character (her eye colour)
glow = Image.new("RGB", (W, H), bg)
gd = ImageDraw.Draw(glow)
gd.ellipse((760, 40, 1240, 620), fill=(min(255, bg[0] + 40), bg[1] + 6, bg[2] + 10))
card = Image.blend(card, glow.filter(ImageFilter.GaussianBlur(120)), 1.0)

# character: ears to waist, scaled to the card's height, on the right
rw, rh = render.size
top, bottom = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.09, 0.66)
crop = render.crop((int(rw * 0.02), int(rh * top), int(rw * 0.98), int(rh * bottom)))
scale = H / crop.height
crop = crop.resize((int(crop.width * scale), H), Image.LANCZOS)
# soften the crop's edges into the card
mask = Image.new("L", crop.size, 255)
md = ImageDraw.Draw(mask)
for i in range(60):
    a = int(255 * i / 60)
    md.rectangle((i, 0, crop.width - 1 - i, crop.height - 1), outline=a)
md.rectangle((0, crop.height - 1, crop.width, crop.height), fill=255)
card.paste(crop, (W - crop.width + 40, 0), mask.filter(ImageFilter.GaussianBlur(6)))

d = ImageDraw.Draw(card)
F = "C:/Windows/Fonts/"
bold = lambda s: ImageFont.truetype(F + "YuGothB.ttc", s)
med = lambda s: ImageFont.truetype(F + "YuGothM.ttc", s)
x = 72
d.text((x, 92), "ANDROID · ON-DEVICE VOICE CHAT", font=med(22), fill=(160, 160, 170))
d.text((x, 128), "orikyara-voice", font=bold(72), fill=(246, 246, 248))
d.rectangle((x, 226, x + 90, 232), fill=(229, 72, 77))
d.text((x, 258), "オリキャラと声で話せるアプリ", font=bold(40), fill=(240, 240, 244))
lines = ["声の合成と音声認識はスマホの中で", "返事は Claude、3D（VRM）で表情も動く", "好感度・記憶・あいさつ・割り込み"]
for i, t in enumerate(lines):
    d.text((x, 336 + i * 44), t, font=med(26), fill=(185, 185, 195))
d.text((x, 534), "github.com/NeonAndKurochan/orikyara-voice", font=med(22), fill=(150, 150, 160))
d.text((x, 566), "MIT License", font=med(20), fill=(120, 120, 130))
card.save(sys.argv[2])
print(sys.argv[2], card.size)
