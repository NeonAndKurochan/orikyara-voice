"""Demo video for SNS: the phone's screen recording (no sound) + ふぇれ子's voice, placed where it was heard.

  <ComfyUI venv python> -I mux_demo.py <screen.mp4> <record start, device epoch ms> <rec dir> <out.mp4> [from_s to_s [crop_top_px]]

crop_top_px cuts the phone's status bar off (130 px on the Nothing Phone (3): clock and icons end at
~100 px, the app's top bar starts at ~180 px).

The rec dir holds turn_<epoch ms>.f32 (48 kHz mono float, written by the app with --ez rec true):
each turn's audio starts at that wall-clock time. The video is cut to [from, to], scaled to fit
1200x1900 (X's portrait limit; 852x1894 for the phone's 9:20 screen) and re-encoded at 30 fps
(H.264, AAC), so it uploads to SNS as is.
"""
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

video_in, start_ms, rec_dir, out_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
cut_from = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
cut_to = float(sys.argv[6]) if len(sys.argv) > 6 else None
crop_top = int(sys.argv[7]) if len(sys.argv) > 7 else 0
RATE, FPS = 48000, 30

src = av.open(video_in)
vin = src.streams.video[0]
duration = float(src.duration / av.time_base) if src.duration else float(vin.duration * vin.time_base)
if cut_to is None or cut_to > duration:
    cut_to = duration
length = cut_to - cut_from
# fit 1200x1900 (X's portrait limit), even sizes
src_w, src_h = vin.codec_context.width, vin.codec_context.height - crop_top
scale = min(1200 / src_w, 1900 / src_h)
W, H = int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2

# her voice, on the recording's timeline
track = np.zeros(int((duration + 1) * RATE), np.float32)
for f in sorted(rec_dir.glob("turn_*.f32")):
    at = int(f.stem.split("_")[1]) - start_ms
    a = np.fromfile(f, "<f4")
    i = int(at * RATE / 1000)
    if i < 0:
        a, i = a[-i:], 0
    n = min(len(a), len(track) - i)
    if n > 0:
        track[i:i + n] += a[:n]
    print(f"{f.name}: at {at / 1000:.2f} s, {len(a) / RATE:.1f} s")
track = np.clip(track[int(cut_from * RATE):int(cut_to * RATE)], -1, 1)

out = av.open(out_path, "w")
vout = out.add_stream("libx264", rate=FPS)
vout.width, vout.height, vout.pix_fmt = W, H, "yuv420p"
vout.options = {"crf": "20", "preset": "medium"}
aout = out.add_stream("aac", rate=RATE, layout="mono")
aout.bit_rate = 160_000

# video: on a 30 fps grid, each output frame shows the latest recorded frame at that time
n_out = int(length * FPS)
k = 0
last = None
for fr in src.decode(vin):
    t = float(fr.pts * vin.time_base)
    while k < n_out and cut_from + k / FPS < t:
        if last is not None:
            last.pts = k
            last.time_base = Fraction(1, FPS)
            for pkt in vout.encode(last):
                out.mux(pkt)
        k += 1
    if t >= cut_from:
        if crop_top:
            a = fr.to_ndarray(format="rgb24")[crop_top:]
            fr = av.VideoFrame.from_ndarray(a, format="rgb24")
        last = fr.reformat(width=W, height=H, format="yuv420p")
    if k >= n_out:
        break
while k < n_out and last is not None:
    last.pts = k
    for pkt in vout.encode(last):
        out.mux(pkt)
    k += 1
for pkt in vout.encode(None):
    out.mux(pkt)

frame_n = 1024
for s in range(0, len(track), frame_n):
    chunk = track[s:s + frame_n]
    if len(chunk) < frame_n:
        chunk = np.pad(chunk, (0, frame_n - len(chunk)))
    af = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="fltp", layout="mono")
    af.sample_rate = RATE
    af.pts = s
    for pkt in aout.encode(af):
        out.mux(pkt)
for pkt in aout.encode(None):
    out.mux(pkt)
out.close()
src.close()
print("->", out_path, f"{length:.1f} s, {W}x{H} {FPS} fps")
