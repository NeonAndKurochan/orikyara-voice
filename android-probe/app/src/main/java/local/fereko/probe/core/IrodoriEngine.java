package local.fereko.probe.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Irodori v4.1-Small-MF, no reference audio, empty caption, ONNX Runtime CPU.
 * Mirrors tools/pc_synth.py (which follows Aratako/Irodori-TTS @ 89f9d8f):
 * text encoder -> duration -> context KV -> 4 MeanFlow DiT steps -> DAC-VAE decode -> tail trim.
 * Text and caption are passed at their real token length; on PC this is bit-identical to 256/512 padding.
 */
public final class IrodoriEngine implements AutoCloseable {
    public static final int SAMPLE_RATE = 48000;
    public static final int HOP = 1920;
    public static final int LATENT_DIM = 32;
    public static final int NUM_LAYERS = 12;
    public static final int NUM_STEPS = 4;
    public static final int TEXT_MAX = 256;
    public static final int CAPTION_MAX = 512;
    public static final int SPEAKER_LEN = 2;
    public static final int SPEAKER_DIM = 768;
    static final double MIN_SECONDS = 0.5, MAX_SECONDS = 30.0;
    static final String[] GRAPHS = {"text_encoder", "duration", "context_kv", "dit_step", "dacvae_decode"};

    /** Frames of latent context on each side of a decode chunk, and the chunk sizes (last one repeats). */
    // 8 frames reproduce the full decode to float precision; 6 frames are ~68 dB SNR (inaudible)
    // and cost less, which matters because the phone decoder is only ~20% faster than real time.
    static final int DECODE_CTX = 6;
    static final int[] CHUNK_PLAN = {8, 16, 32, 64};

    /** Which graphs run on the XNNPACK execution provider (the rest stay on the default CPU provider). */
    public enum Accel { NONE, DECODER, ALL }

    public interface Listener {
        void stage(String name, long nanos);
        default boolean cancelled() { return false; }
    }

    /**
     * Receives decoded audio in order, as soon as each chunk is ready.
     * {@code plan} holds one {start, end, windowStart, windowEnd} frame range per chunk.
     */
    public interface AudioSink {
        void chunk(float[] samples, int chunkIndex, List<int[]> plan, long decodeNanosSoFar);
    }

    /** Chunk ranges for decoding {@code total} output frames out of a latent of {@code latentFrames}. */
    public static List<int[]> decodePlan(int total, int latentFrames) {
        List<int[]> plan = new ArrayList<>();
        for (int st = 0, idx = 0; st < total; idx++) {
            int en = Math.min(total, st + CHUNK_PLAN[Math.min(idx, CHUNK_PLAN.length - 1)]);
            plan.add(new int[]{st, en, Math.max(0, st - DECODE_CTX), Math.min(latentFrames, en + DECODE_CTX)});
            st = en;
        }
        return plan;
    }

    public static final class Result {
        public String normalized;
        public long[] tokenIds;
        public double predFrames;
        public int latentFrames;
        public int flattenFrame;
        public float[] latent;      // [frames * 32]
        public float[] audio;       // trimmed mono, 48 kHz
        public final Map<String, Long> stageNanos = new LinkedHashMap<>();
    }

    public static final class CancelledException extends RuntimeException {
        public CancelledException() { super("cancelled"); }
    }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final Map<String, OrtSession> sessions = new HashMap<>();
    public final Map<String, Long> loadNanos = new LinkedHashMap<>();
    private final UnigramTokenizer tokenizer;
    private final float[] noRefSpeaker;
    /** Registered voice (speaker.f32 / speaker.json from fereko-pack-voice, or a character's own), or null. */
    private volatile float[] refSpeaker;
    private volatile int refSpeakerLen;
    private volatile boolean useReference = true;
    private final int bosId, padId;

    public IrodoriEngine(File packDir, int threads) throws IOException, OrtException {
        this(packDir, threads, Accel.NONE);
    }

    public IrodoriEngine(File packDir, int threads, Accel accel) throws IOException, OrtException {
        this(packDir, threads, accel, "");
    }

    /**
     * @param ditVariant "" for the FP32 DiT, or the suffix of a quantized file next to it
     *                   (e.g. "int8dyn" loads dit_step.int8dyn.onnx).
     */
    public IrodoriEngine(File packDir, int threads, Accel accel, String ditVariant) throws IOException, OrtException {
        this(packDir, threads, accel, ditVariant, "");
    }

    /** @param decVariant "" for the FP32 decoder, or a suffix such as "fp16" (dacvae_decode.fp16.onnx). */
    public IrodoriEngine(File packDir, int threads, Accel accel, String ditVariant, String decVariant) throws IOException, OrtException {
        long t0 = System.nanoTime();
        Map<String, Object> pack = MiniJson.obj(MiniJson.parse(read(new File(packDir, "pack.json"))));
        bosId = ((Double) pack.get("bos_id")).intValue();
        padId = ((Double) pack.get("pad_id")).intValue();
        tokenizer = new UnigramTokenizer(read(new File(packDir, "tokenizer.json")));
        noRefSpeaker = readF32(new File(packDir, "noref_speaker.f32"));
        if (noRefSpeaker.length != SPEAKER_LEN * SPEAKER_DIM) throw new IOException("noref_speaker.f32 size");
        setReference(packDir);
        loadNanos.put("tokenizer", System.nanoTime() - t0);
        try {
            for (String g : GRAPHS) {
                long t = System.nanoTime();
                OrtSession.SessionOptions so = new OrtSession.SessionOptions();
                so.setInterOpNumThreads(1);
                boolean xnn = accel == Accel.ALL || (accel == Accel.DECODER && g.equals("dacvae_decode"));
                if (xnn) {
                    // XNNPACK uses its own thread pool; ORT's pool then only runs the ops XNNPACK doesn't take.
                    so.setIntraOpNumThreads(1);
                    so.addConfigEntry("session.intra_op.allow_spinning", "0");
                    Map<String, String> xo = new HashMap<>();
                    xo.put("intra_op_num_threads", String.valueOf(threads));
                    so.addXnnpack(xo);
                } else {
                    so.setIntraOpNumThreads(threads);
                }
                String variant = g.equals("dit_step") ? ditVariant : g.equals("dacvae_decode") ? decVariant : "";
                String file = variant.isEmpty() ? g + ".onnx" : g + "." + variant + ".onnx";
                so.setSessionLogLevel(ai.onnxruntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
                sessions.put(g, env.createSession(new File(packDir, file).getAbsolutePath(), so));
                loadNanos.put(g, System.nanoTime() - t);
            }
        } catch (OrtException | RuntimeException e) {
            close();
            throw e;
        }
    }

    public UnigramTokenizer tokenizer() { return tokenizer; }

    /** True when a registered voice is installed (fereko-pack-voice). */
    public boolean hasReference() { return refSpeaker != null; }

    /** Use the registered voice when installed (default), or the random no-reference voice. */
    public void setUseReference(boolean use) { useReference = use; }

    /**
     * The registered voice from speaker.f32 + speaker.json in this folder (the pack for ふぇれ子,
     * a character's own folder for others); none if the folder has no speaker.f32.
     */
    public void setReference(File dir) throws IOException {
        File refFile = new File(dir, "speaker.f32");
        if (!refFile.isFile()) {
            refSpeaker = null;
            refSpeakerLen = 0;
            return;
        }
        Map<String, Object> meta = MiniJson.obj(MiniJson.parse(read(new File(dir, "speaker.json"))));
        int len = ((Double) meta.get("len")).intValue();
        float[] spk = readF32(refFile);
        if (spk.length != len * SPEAKER_DIM) throw new IOException("speaker.f32 size");
        refSpeaker = spk;
        refSpeakerLen = len;
    }

    /**
     * A decoder with a fixed input length (e.g. LiteRT on the GPU, which needs static shapes):
     * takes window() latent frames ([window x 32], frame-major), returns window() * HOP samples.
     */
    public interface WindowDecoder extends AutoCloseable {
        /** Supported window sizes, ascending. The first chunk uses the smallest (sooner first sound). */
        int[] windows();
        float[] decode(float[] latentWindow, int window) throws Exception;
        @Override void close();
    }

    private volatile WindowDecoder windowDecoder;

    /** Decode with this fixed-window decoder instead of the ONNX one (null = back to ONNX). */
    public void setWindowDecoder(WindowDecoder d) { windowDecoder = d; }

    /**
     * Speaking rate: the predicted length is divided by this (1.15 = 15% shorter, so faster).
     * The model fills whatever length it is given, so this changes the tempo, not the pitch.
     */
    public volatile float speed = 1f;

    public WindowDecoder windowDecoder() { return windowDecoder; }

    public int padId() { return padId; }

    /** @param noise initial latent noise [frames*32], or null to draw N(0,1) from {@code seed}. */
    public Result synthesize(String text, float[] noise, long seed, Listener l) throws OrtException {
        return synthesize(text, noise, seed, l, null);
    }

    public Result synthesize(String text, float[] noise, long seed, Listener l, AudioSink sink) throws OrtException {
        return synthesize(text, noise, seed, l, sink, "");
    }

    /**
     * @param caption voice description (Irodori "caption"), e.g. "明るくかわいい、若い女性の声"; "" for none.
     *                Without one the voice is random per sentence and sometimes comes out male.
     */
    public Result synthesize(String text, float[] noise, long seed, Listener l, AudioSink sink, String caption) throws OrtException {
        Result r = new Result();
        caption = TextNormalizer.pyStrip(caption == null ? "" : caption);   // upstream strips the caption, no normalize_text
        long t = System.nanoTime();
        r.normalized = TextNormalizer.normalize(text);
        if (r.normalized.isEmpty()) throw new IllegalArgumentException("text became empty after normalization");
        r.tokenIds = tokenizer.encodeWithBos(r.normalized, bosId, TEXT_MAX);
        stage(r, l, "tokenize", t);

        int tl = r.tokenIds.length;
        float[] textMask = filled(tl, 1f);
        boolean hasCaption = !caption.isEmpty();
        long[] capTokens = hasCaption ? tokenizer.encodeWithBos(caption, bosId, CAPTION_MAX) : new long[]{bosId};
        float[] capMask = hasCaption ? filled(capTokens.length, 1f) : new float[1];   // empty caption: [BOS], mask zeroed
        // Registered voice: its precomputed speaker state, all positions valid. Otherwise the
        // no-reference state with everything masked (inference_runtime._load_reference_latent).
        boolean ref = useReference && refSpeaker != null;
        float[] spk = ref ? refSpeaker : noRefSpeaker;
        int spkLen = ref ? refSpeakerLen : SPEAKER_LEN;
        float[] spkMask = ref ? filled(spkLen, 1f) : new float[SPEAKER_LEN];
        try (OnnxTensor textIds = OnnxTensor.createTensor(env, LongBuffer.wrap(r.tokenIds), new long[]{1, tl});
             OnnxTensor textMaskT = f32(textMask, 1, tl);
             OnnxTensor capIds = OnnxTensor.createTensor(env, LongBuffer.wrap(capTokens), new long[]{1, capTokens.length});
             OnnxTensor capMaskT = f32(capMask, 1, capMask.length);
             OnnxTensor hasCap = f32(new float[]{hasCaption ? 1f : 0f}, 1);
             OnnxTensor spkState = f32(spk, 1, spkLen, SPEAKER_DIM);
             OnnxTensor spkMaskT = f32(spkMask, 1, spkLen);
             OnnxTensor hasSpk = f32(new float[]{ref ? 1f : 0f}, 1);
             OnnxTensor kvScales = f32(filled(NUM_LAYERS, 1f), NUM_LAYERS)) {

            t = System.nanoTime();
            try (OrtSession.Result te = run("text_encoder", map("input_ids", textIds, "mask", textMaskT));
                 OrtSession.Result ce = run("text_encoder", map("input_ids", capIds, "mask", capMaskT))) {
                OnnxTensor textState = (OnnxTensor) te.get("text_state").get();
                OnnxTensor capState = (OnnxTensor) ce.get("caption_state").get();
                stage(r, l, "text_encoder", t);

                t = System.nanoTime();
                float logFrames;
                Map<String, OnnxTensor> dIn = map("text_state", textState, "text_mask", textMaskT,
                        "speaker_state", spkState, "has_speaker", hasSpk);
                dIn.putAll(map("caption_state", capState, "caption_mask", capMaskT, "has_caption", hasCap));
                try (OrtSession.Result d = run("duration", dIn)) {
                    logFrames = ((float[]) ((OnnxTensor) d.get("log_frames").get()).getValue())[0];
                }
                r.predFrames = Math.expm1(logFrames);
                int minF = Math.max(1, (int) Math.ceil(MIN_SECONDS * SAMPLE_RATE / HOP));
                int maxF = Math.max(1, (int) Math.floor(MAX_SECONDS * SAMPLE_RATE / HOP));
                int frames = Math.max(minF, Math.min(maxF, (int) Math.rint(r.predFrames / speed)));  // Python round(): half-even
                r.latentFrames = frames;
                stage(r, l, "duration", t);

                t = System.nanoTime();
                try (OrtSession.Result kv = run("context_kv",
                        map("text_state", textState, "speaker_state", spkState, "caption_state", capState))) {
                    stage(r, l, "context_kv", t);

                    float[] x;
                    if (noise != null) {
                        if (noise.length != frames * LATENT_DIM)
                            throw new IllegalArgumentException("noise frames " + noise.length / LATENT_DIM + " != " + frames);
                        x = noise.clone();
                    } else {
                        x = new float[frames * LATENT_DIM];
                        Random rng = new Random(seed);
                        for (int i = 0; i < x.length; i++) x[i] = (float) rng.nextGaussian();
                    }

                    Map<String, OnnxTensor> base = new HashMap<>();
                    for (Map.Entry<String, ai.onnxruntime.OnnxValue> e : kv) base.put(e.getKey(), (OnnxTensor) e.getValue());
                    base.put("text_mask", textMaskT);
                    base.put("speaker_mask", spkMaskT);
                    base.put("caption_mask", capMaskT);
                    base.put("speaker_kv_scales", kvScales);

                    for (int step = 0; step < NUM_STEPS; step++) {
                        if (l != null && l.cancelled()) throw new CancelledException();
                        t = System.nanoTime();
                        float tv = 1f - step / (float) NUM_STEPS;
                        float nx = 1f - (step + 1) / (float) NUM_STEPS;
                        Map<String, OnnxTensor> in = new HashMap<>(base);
                        try (OnnxTensor xt = f32(x, 1, frames, LATENT_DIM);
                             OnnxTensor tt = f32(new float[]{tv}, 1);
                             OnnxTensor dt = f32(new float[]{tv - nx}, 1)) {
                            in.put("x_t", xt);
                            in.put("t", tt);
                            in.put("delta_t", dt);
                            try (OrtSession.Result v = run("dit_step", in)) {
                                FloatBuffer vb = ((OnnxTensor) v.get("v_pred").get()).getFloatBuffer();
                                float h = nx - tv;
                                for (int i = 0; i < x.length; i++) x[i] += vb.get(i) * h;
                            }
                        }
                        stage(r, l, "dit_step" + step, t);
                    }
                    r.latent = x;
                }
            }
        }

        // Tail trim is decided from the full latent, so it is known before decoding starts.
        r.flattenFrame = findFlatteningPoint(r.latent, r.latentFrames, 20, 0.05, 0.1);
        int maxSamples = r.latentFrames * HOP;
        if ((long) r.flattenFrame * HOP > 0) maxSamples = Math.min(maxSamples, r.flattenFrame * HOP);
        r.audio = new float[maxSamples];
        decodeChunked(r, maxSamples, l, sink);
        return r;
    }

    /**
     * Decodes the latent in growing chunks with DECODE_CTX frames of context on each side.
     * On PC, 8 context frames reproduce the full decode to float precision (SNR ~131 dB), so the
     * audio is the same as one full decode; the first chunk is ready much earlier.
     */
    private void decodeChunked(Result r, int maxSamples, Listener l, AudioSink sink) throws OrtException {
        WindowDecoder wd = windowDecoder;
        if (wd != null) {
            decodeFixed(wd, r, maxSamples, l, sink);
            return;
        }
        int total = (maxSamples + HOP - 1) / HOP;
        List<int[]> plan = decodePlan(total, r.latentFrames);
        long decodeNs = 0;
        int written = 0;
        for (int idx = 0; idx < plan.size(); idx++) {
            if (l != null && l.cancelled()) throw new CancelledException();
            long t = System.nanoTime();
            int[] c = plan.get(idx);
            int st = c[0], en = c[1], a = c[2], b = c[3];
            float[] window = new float[(b - a) * LATENT_DIM];
            System.arraycopy(r.latent, a * LATENT_DIM, window, 0, window.length);
            int from = (st - a) * HOP;
            int n = Math.min((en - st) * HOP, maxSamples - st * HOP);
            float[] part = new float[n];
            try (OnnxTensor lat = f32(window, 1, b - a, LATENT_DIM);
                 OrtSession.Result d = run("dacvae_decode", map("latent", lat))) {
                FloatBuffer wb = ((OnnxTensor) d.get("waveform").get()).getFloatBuffer();
                wb.position(from);
                wb.get(part);
            }
            System.arraycopy(part, 0, r.audio, written, n);
            written += n;
            long ns = System.nanoTime() - t;
            decodeNs += ns;
            r.stageNanos.merge("dacvae_decode", ns, Long::sum);
            if (idx == 0) r.stageNanos.put("decode_first_chunk", ns);
            if (l != null) l.stage("dacvae_decode " + en + "/" + total, ns);
            if (sink != null) sink.chunk(part, idx, plan, decodeNs);
        }
    }

    /**
     * Fixed windows of W frames: W - 2*DECODE_CTX new frames each, DECODE_CTX frames of context on
     * each side, zero latent frames outside the utterance. Interior output matches the ONNX decode;
     * the first/last ~0.24 s differ slightly (about -48 dBFS on PC) because of the zero padding.
     */
    private void decodeFixed(WindowDecoder wd, Result r, int maxSamples, Listener l, AudioSink sink) {
        int total = (maxSamples + HOP - 1) / HOP;
        int[] ws = wd.windows();
        List<int[]> plan = new ArrayList<>();
        for (int st = 0, idx = 0; st < total; idx++) {
            int w = idx == 0 ? ws[0] : ws[ws.length - 1];
            int step = w - 2 * DECODE_CTX;
            plan.add(new int[]{st, Math.min(total, st + step), st - DECODE_CTX, st - DECODE_CTX + w});
            st += step;
        }
        long decodeNs = 0;
        int written = 0;
        for (int idx = 0; idx < plan.size(); idx++) {
            if (l != null && l.cancelled()) throw new CancelledException();
            long t = System.nanoTime();
            int[] c = plan.get(idx);
            int st = c[0], en = c[1], a = c[2];
            int w = c[3] - c[2];
            float[] window = new float[w * LATENT_DIM];
            int lo = Math.max(0, a), hi = Math.min(r.latentFrames, a + w);
            System.arraycopy(r.latent, lo * LATENT_DIM, window, (lo - a) * LATENT_DIM, (hi - lo) * LATENT_DIM);
            float[] out;
            try {
                out = wd.decode(window, w);
            } catch (Exception e) {
                throw new IllegalStateException("window decoder failed: " + e, e);
            }
            int from = (st - a) * HOP;
            int n = Math.min((en - st) * HOP, maxSamples - st * HOP);
            float[] part = new float[n];
            System.arraycopy(out, from, part, 0, n);
            System.arraycopy(part, 0, r.audio, written, n);
            written += n;
            long ns = System.nanoTime() - t;
            decodeNs += ns;
            r.stageNanos.merge("dacvae_decode", ns, Long::sum);
            if (idx == 0) r.stageNanos.put("decode_first_chunk", ns);
            if (l != null) l.stage("dacvae_decode " + en + "/" + total, ns);
            if (sink != null) sink.chunk(part, idx, plan, decodeNs);
        }
    }

    /** inference_runtime.find_flattening_point on a (T, D) latent, zero-padded by one window. */
    static int findFlatteningPoint(float[] lat, int frames, int window, double stdThr, double meanThr) {
        int d = LATENT_DIM;
        for (int i = 0; i < frames; i++) {
            double sum = 0, sq = 0;
            int cnt = window * d;
            for (int f = i; f < i + window && f < frames; f++) {
                for (int k = 0; k < d; k++) { double v = lat[f * d + k]; sum += v; sq += v * v; }
            }
            double mean = sum / cnt;
            double var = Math.max(0, sq / cnt - mean * mean);
            if (Math.sqrt(var) < stdThr && Math.abs(mean) < meanThr) return i;
        }
        return frames;
    }

    private OrtSession.Result run(String g, Map<String, OnnxTensor> in) throws OrtException {
        return sessions.get(g).run(in);
    }

    private static void stage(Result r, Listener l, String name, long t0) {
        long ns = System.nanoTime() - t0;
        r.stageNanos.merge(name, ns, Long::sum);
        if (l != null) l.stage(name, ns);
    }

    private OnnxTensor f32(float[] a, long... shape) throws OrtException {
        FloatBuffer b = ByteBuffer.allocateDirect(a.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        b.put(a).rewind();
        return OnnxTensor.createTensor(env, b, shape);
    }

    private static Map<String, OnnxTensor> map(Object... kv) {
        Map<String, OnnxTensor> m = new HashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], (OnnxTensor) kv[i + 1]);
        return m;
    }

    private static float[] filled(int n, float v) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, v);
        return a;
    }

    public static String read(File f) throws IOException {
        return new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);
    }

    public static float[] readF32(File f) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(f.toPath())).order(ByteOrder.LITTLE_ENDIAN);
        float[] a = new float[b.remaining() / 4];
        b.asFloatBuffer().get(a);
        return a;
    }

    @Override
    public void close() {
        for (OrtSession s : sessions.values()) {
            try { s.close(); } catch (OrtException ignored) { }
        }
        sessions.clear();
    }
}
