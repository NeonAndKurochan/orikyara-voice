package local.fereko.probe.core;

/**
 * Kaldi-compatible log-mel filterbank, matching what sherpa-onnx (kaldi-native-fbank) feeds the
 * ReazonSpeech k2 v2 model: 16 kHz, 25 ms frames every 10 ms, 80 mel bins from 20 Hz to 7600 Hz,
 * remove DC, pre-emphasis 0.97, povey window, 512-point FFT, power spectrum, natural log,
 * no dither, snip_edges=false (frames are centred; edges reflected). Samples are in [-1, 1].
 */
public final class Fbank {
    public static final int RATE = 16000, BINS = 80;
    static final int LEN = 400, SHIFT = 160, NFFT = 512;
    private static final float PREEMPH = 0.97f;
    private static final float EPS = 1.1920929e-07f;   // FLT_EPSILON, kaldi's log floor

    private final float[] window = new float[LEN];
    private final float[][] mel = new float[BINS][];   // weights over FFT bins melStart[b]..
    private final int[] melStart = new int[BINS];
    private final double[] cos = new double[NFFT / 2], sin = new double[NFFT / 2];
    private final int[] rev = new int[NFFT];

    public Fbank() {
        for (int i = 0; i < LEN; i++)
            window[i] = (float) Math.pow(0.5 - 0.5 * Math.cos(2 * Math.PI * i / (LEN - 1)), 0.85);
        int nBins = NFFT / 2;
        double binHz = (double) RATE / NFFT;
        double lo = melOf(20), hi = melOf(RATE / 2.0 - 400), delta = (hi - lo) / (BINS + 1);
        for (int b = 0; b < BINS; b++) {
            double left = lo + b * delta, center = left + delta, right = center + delta;
            int first = -1, last = -1;
            float[] w = new float[nBins];
            for (int i = 0; i < nBins; i++) {
                double m = melOf(binHz * i);
                if (m > left && m < right) {
                    w[i] = (float) (m <= center ? (m - left) / (center - left) : (right - m) / (right - center));
                    if (first < 0) first = i;
                    last = i;
                }
            }
            melStart[b] = Math.max(first, 0);
            mel[b] = java.util.Arrays.copyOfRange(w, melStart[b], last + 1);
        }
        for (int i = 0; i < NFFT / 2; i++) {
            cos[i] = Math.cos(-2 * Math.PI * i / NFFT);
            sin[i] = Math.sin(-2 * Math.PI * i / NFFT);
        }
        int bits = Integer.numberOfTrailingZeros(NFFT);
        for (int i = 0; i < NFFT; i++) rev[i] = Integer.reverse(i) >>> (32 - bits);
    }

    private static double melOf(double hz) { return 1127.0 * Math.log(1 + hz / 700.0); }

    public static int numFrames(int samples) { return (samples + SHIFT / 2) / SHIFT; }

    /** @return frames x BINS, row-major */
    public float[] compute(float[] x, int n) {
        int frames = numFrames(n);
        float[] out = new float[frames * BINS];
        float[] w = new float[LEN];
        double[] re = new double[NFFT], im = new double[NFFT];
        for (int f = 0; f < frames; f++) {
            int start = f * SHIFT + SHIFT / 2 - LEN / 2;
            double mean = 0;
            for (int i = 0; i < LEN; i++) {
                int s = start + i;
                while (s < 0 || s >= n) s = s < 0 ? -s - 1 : 2 * n - 1 - s;
                w[i] = x[s];
                mean += w[i];
            }
            mean /= LEN;
            for (int i = 0; i < LEN; i++) w[i] -= (float) mean;
            for (int i = LEN - 1; i > 0; i--) w[i] -= PREEMPH * w[i - 1];
            w[0] -= PREEMPH * w[0];
            java.util.Arrays.fill(im, 0);
            for (int i = 0; i < NFFT; i++) {
                int j = rev[i];
                re[i] = j < LEN ? w[j] * window[j] : 0;
            }
            fft(re, im);
            for (int b = 0; b < BINS; b++) {
                float[] mw = mel[b];
                int s = melStart[b];
                double e = 0;
                for (int k = 0; k < mw.length; k++) {
                    int i = s + k;
                    e += mw[k] * (re[i] * re[i] + im[i] * im[i]);
                }
                out[f * BINS + b] = (float) Math.log(Math.max(e, EPS));
            }
        }
        return out;
    }

    /** In-place radix-2 FFT; input already in bit-reversed order. */
    private void fft(double[] re, double[] im) {
        for (int size = 2; size <= NFFT; size <<= 1) {
            int half = size >> 1, step = NFFT / size;
            for (int s = 0; s < NFFT; s += size) {
                for (int k = 0; k < half; k++) {
                    double c = cos[k * step], si = sin[k * step];
                    int a = s + k, b = a + half;
                    double tr = re[b] * c - im[b] * si, ti = re[b] * si + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }
}
