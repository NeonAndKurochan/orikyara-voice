"""Which side needs context? Decode chunks with (left, right) context and compare with the full decode."""
import sys
import numpy as np, onnxruntime as ort
HOP = 1920
lat = np.load(sys.argv[2])
so = ort.SessionOptions(); so.intra_op_num_threads = 6
s = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
dec = lambda x: s.run(None, {"latent": np.ascontiguousarray(x)})[0][0, 0]
full = dec(lat); T = lat.shape[1]
def run(plan, L, R):
    out = np.zeros_like(full); st = 0; i = 0; frames = 0
    while st < T:
        en = min(T, st + plan[min(i, len(plan) - 1)]); a = max(0, st - L); b = min(T, en + R)
        w = dec(lat[:, a:b]); frames += b - a
        out[st*HOP:en*HOP] = w[(st-a)*HOP:(st-a)*HOP + (en-st)*HOP]; st = en; i += 1
    err = out - full
    return 10*np.log10((full**2).sum()/max((err**2).sum(), 1e-20)), np.abs(err).max(), frames
for L, R in [(8,8),(0,8),(8,0),(2,6),(4,4),(4,6),(6,4),(3,5),(5,5),(6,6)]:
    snr, md, fr = run([16], L, R)
    print(f"L{L} R{R}: SNR {snr:6.1f} dB maxdiff {md:.4f}  decoded {fr} frames for {T}")
for plan in ([8,16,32],[8,16,32,64],[8,24,64],[12,24,48,96]):
    snr, md, fr = run(plan, 8, 8)
    print(f"plan {plan} ctx 8/8: decoded {fr} frames for {T} ({fr/T:.2f}x)")
