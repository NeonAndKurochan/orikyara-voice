"""Does decoding the latent in chunks (with context frames on both sides) match a full decode?"""
import sys, time
import numpy as np, onnxruntime as ort
HOP = 1920
lat = np.load(sys.argv[2])  # (1, T, 32)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
s = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
dec = lambda x: s.run(None, {"latent": np.ascontiguousarray(x)})[0][0, 0]
full = dec(lat); T = lat.shape[1]
t0 = time.perf_counter(); dec(lat); tf = time.perf_counter() - t0
print(f"full: {T} frames, {tf*1000:.0f} ms (4 threads)")
for chunk in (12, 25):
    for ctx in (0, 2, 4, 8, 16):
        out = np.zeros_like(full); t_first = None; t0 = time.perf_counter()
        for st in range(0, T, chunk):
            en = min(T, st + chunk); a = max(0, st - ctx); b = min(T, en + ctx)
            w = dec(lat[:, a:b])
            out[st*HOP:en*HOP] = w[(st-a)*HOP:(st-a)*HOP + (en-st)*HOP]
            if t_first is None: t_first = time.perf_counter() - t0
        tt = time.perf_counter() - t0
        err = out - full
        snr = 10*np.log10((full**2).sum() / max((err**2).sum(), 1e-20))
        print(f"chunk {chunk:2d} ctx {ctx:2d}: SNR {snr:6.1f} dB  maxdiff {np.abs(err).max():.4f}  first chunk {t_first*1000:4.0f} ms  total {tt*1000:5.0f} ms")
