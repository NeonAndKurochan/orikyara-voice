package local.fereko.probe.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Collections;

/**
 * Japanese speech recognition on the device: ReazonSpeech k2 v2 (non-streaming zipformer2
 * transducer, Apache-2.0) with greedy search, the same as sherpa-onnx's "greedy_search".
 * Files in dir: encoder/decoder/joiner-epoch-99-avg-1.int8.onnx and tokens.txt.
 */
public final class ReazonAsr implements AutoCloseable {
    private static final int CONTEXT = 2, BLANK = 0, DIM = 512;

    public static final class Result {
        public String text;
        public int[] ids;
        public long featNanos, encNanos, searchNanos;
        public double audioSec;
    }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession enc, dec, join;
    private final String[] vocab;
    private final Fbank fbank = new Fbank();
    public final long loadNanos;

    public ReazonAsr(File dir, int threads) throws IOException, OrtException {
        long t = System.nanoTime();
        String[] lines = IrodoriEngine.read(new File(dir, "tokens.txt")).split("\n");
        int max = 0;
        for (String l : lines) if (!l.isEmpty()) max = Math.max(max, Integer.parseInt(l.substring(l.lastIndexOf('\t') + 1).trim()));
        vocab = new String[max + 1];
        for (String l : lines) {
            if (l.isEmpty()) continue;
            int tab = l.lastIndexOf('\t');
            vocab[Integer.parseInt(l.substring(tab + 1).trim())] = l.substring(0, tab);
        }
        enc = session(new File(dir, "encoder-epoch-99-avg-1.int8.onnx"), threads);
        dec = session(new File(dir, "decoder-epoch-99-avg-1.int8.onnx"), 1);
        join = session(new File(dir, "joiner-epoch-99-avg-1.int8.onnx"), 1);
        loadNanos = System.nanoTime() - t;
    }

    private OrtSession session(File f, int threads) throws OrtException {
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setInterOpNumThreads(1);
        so.setIntraOpNumThreads(threads);
        so.setSessionLogLevel(ai.onnxruntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
        return env.createSession(f.getAbsolutePath(), so);
    }

    /** @param pcm 16 kHz mono samples in [-1, 1] */
    public Result recognize(float[] pcm, int n) throws OrtException {
        Result r = new Result();
        r.audioSec = n / (double) Fbank.RATE;
        long t = System.nanoTime();
        float[] feats = fbank.compute(pcm, n);
        r.featNanos = System.nanoTime() - t;
        return recognizeFeatures(feats, r);
    }

    public Result recognizeFeatures(float[] feats, Result r) throws OrtException {
        if (r == null) r = new Result();
        int frames = feats.length / Fbank.BINS;
        long t = System.nanoTime();
        float[] encOut;
        int encFrames;
        try (OnnxTensor x = OnnxTensor.createTensor(env, FloatBuffer.wrap(feats), new long[]{1, frames, Fbank.BINS});
             OnnxTensor xl = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{frames}), new long[]{1});
             OrtSession.Result o = enc.run(java.util.Map.of("x", x, "x_lens", xl))) {
            encFrames = (int) ((long[]) o.get(1).getValue())[0];
            FloatBuffer fb = ((OnnxTensor) o.get(0)).getFloatBuffer();
            encOut = new float[fb.remaining()];
            fb.get(encOut);
        }
        r.encNanos = System.nanoTime() - t;

        t = System.nanoTime();
        int[] hyp = new int[encFrames + CONTEXT];
        int len = CONTEXT;   // hyp[0..1] = blank context
        float[] decOut = decode(hyp, len);
        float[] eo = new float[DIM];
        for (int f = 0; f < encFrames; f++) {
            System.arraycopy(encOut, f * DIM, eo, 0, DIM);
            int k = joinArgmax(eo, decOut);
            if (k != BLANK) {
                hyp[len++] = k;
                decOut = decode(hyp, len);
            }
        }
        r.searchNanos = System.nanoTime() - t;
        r.ids = java.util.Arrays.copyOfRange(hyp, CONTEXT, len);
        StringBuilder sb = new StringBuilder();
        for (int id : r.ids) sb.append(vocab[id]);
        r.text = sb.toString().replace('▁', ' ').trim();
        return r;
    }

    private float[] decode(int[] hyp, int len) throws OrtException {
        long[] y = new long[CONTEXT];
        for (int i = 0; i < CONTEXT; i++) y[i] = hyp[len - CONTEXT + i];
        try (OnnxTensor yt = OnnxTensor.createTensor(env, LongBuffer.wrap(y), new long[]{1, CONTEXT});
             OrtSession.Result o = dec.run(Collections.singletonMap("y", yt))) {
            FloatBuffer fb = ((OnnxTensor) o.get(0)).getFloatBuffer();
            float[] d = new float[DIM];
            fb.get(d);
            return d;
        }
    }

    private int joinArgmax(float[] encFrame, float[] decOut) throws OrtException {
        try (OnnxTensor e = OnnxTensor.createTensor(env, FloatBuffer.wrap(encFrame), new long[]{1, DIM});
             OnnxTensor d = OnnxTensor.createTensor(env, FloatBuffer.wrap(decOut), new long[]{1, DIM});
             OrtSession.Result o = join.run(java.util.Map.of("encoder_out", e, "decoder_out", d))) {
            FloatBuffer fb = ((OnnxTensor) o.get(0)).getFloatBuffer();
            int best = 0;
            float bv = Float.NEGATIVE_INFINITY;
            for (int i = 0; fb.hasRemaining(); i++) {
                float v = fb.get();
                if (v > bv) { bv = v; best = i; }
            }
            return best;
        }
    }

    @Override public void close() throws OrtException {
        enc.close();
        dec.close();
        join.close();
    }
}
