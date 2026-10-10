"""Reference for the app's own ASR: kaldi-native-fbank + onnxruntime greedy transducer search,
compared with sherpa-onnx. Finds the feature options sherpa-onnx uses for ReazonSpeech k2 v2 and
writes fixtures (input samples, fbank, token ids, text) for the Java port.

  python -I asr_ref.py <wav> [<fixture dir>]
"""
import sys
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import sherpa_onnx
import soundfile as sf

M = Path(__file__).resolve().parent.parent / "models/reazonspeech-k2-v2"


def fbank(audio16k: np.ndarray, scale: float) -> np.ndarray:
    o = knf.FbankOptions()
    o.frame_opts.dither = 0
    o.frame_opts.snip_edges = False
    o.frame_opts.samp_freq = 16000
    o.mel_opts.num_bins = 80
    o.mel_opts.high_freq = -400
    f = knf.OnlineFbank(o)
    f.accept_waveform(16000, (audio16k * scale).tolist())
    f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


def load_vocab():
    v = {}
    for line in (M / "tokens.txt").read_text(encoding="utf-8").splitlines():
        sym, i = line.rsplit("\t", 1)
        v[int(i)] = sym
    return v


class Greedy:
    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.enc = ort.InferenceSession(str(M / "encoder-epoch-99-avg-1.int8.onnx"), so)
        self.dec = ort.InferenceSession(str(M / "decoder-epoch-99-avg-1.int8.onnx"), so)
        self.join = ort.InferenceSession(str(M / "joiner-epoch-99-avg-1.int8.onnx"), so)

    def __call__(self, feats: np.ndarray):
        x = feats[None]
        eo, el = self.enc.run(None, {"x": x, "x_lens": np.array([x.shape[1]], np.int64)})
        hyp = [0, 0]   # blank context (context_size 2)
        dout = self.dec.run(None, {"y": np.array([hyp[-2:]], np.int64)})[0]
        for t in range(int(el[0])):
            logit = self.join.run(None, {"encoder_out": eo[:, t], "decoder_out": dout})[0]
            k = int(logit[0].argmax())
            if k != 0:
                hyp.append(k)
                dout = self.dec.run(None, {"y": np.array([hyp[-2:]], np.int64)})[0]
        return hyp[2:]


def main():
    wav = sys.argv[1]
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        assert sr % 16000 == 0, sr
        d = sr // 16000   # windowed-sinc low-pass, then decimate
        n = np.arange(-64 * d, 64 * d + 1)
        h = np.sinc(n * 0.9 / d) * 0.9 / d * np.hamming(len(n))
        audio = np.convolve(audio, h, mode="same")[::d].astype(np.float32)
        sr = 16000
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(M / "encoder-epoch-99-avg-1.int8.onnx"), decoder=str(M / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(M / "joiner-epoch-99-avg-1.int8.onnx"), tokens=str(M / "tokens.txt"),
        num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
    s = rec.create_stream()
    s.accept_waveform(sr, audio)
    rec.decode_stream(s)
    print("sherpa :", s.result.text)
    vocab = load_vocab()
    g = Greedy()
    for scale in [1.0, 32768.0]:
        ids = g(fbank(audio, scale))
        text = "".join(vocab[i] for i in ids).replace("▁", " ").strip()
        print(f"x{scale:<7g}:", text)
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        f = fbank(audio, scale)
        ids = g(f)
        audio.astype("<f4").tofile(out / "audio16k.f32")
        f.astype("<f4").tofile(out / "fbank.f32")
        (out / "expected.txt").write_text(
            f"{f.shape[0]}\n{' '.join(map(str, ids))}\n{''.join(vocab[i] for i in ids)}\n", encoding="utf-8")
        print("fixture ->", out, f.shape)


main()
