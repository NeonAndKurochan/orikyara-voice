package local.fereko.probe;

import android.app.Activity;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * NPU tests in their own process (android:process=":npu"): a QNN call that hangs in native code
 * cannot be interrupted, so 強制終了 kills this process and the main screens keep working.
 * Every step is written to runs.jsonl *before* it starts, so a hang shows where it stopped.
 */
public class NpuActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private TextView log, status;
    private ScrollView scroll;
    private long t0;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        int pad = Math.round(10 * getResources().getDisplayMetrics().density);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);

        LinearLayout row = new LinearLayout(this);
        row.addView(button("情報", this::info));
        row.addView(button("小ctx", () -> tinyTest(true)));
        row.addView(button("小online", () -> tinyTest(false)));
        row.addView(button("c2d試験", () -> test("htp-c2d")));
        // one button per EP-context model in the pack (e.g. compiled on PC for v75 / v79)
        File[] ctx = new File(getFilesDir(), "pack").listFiles((d, n) -> (n.startsWith("dacvae") || n.startsWith("npu_E")) && n.endsWith(".qnnctx.onnx")
                && !n.startsWith("dacvae_decode.static44c2d.qnnctx") && !n.contains(".v75.") && !n.contains(".v79."));
        LinearLayout ctxRow = new LinearLayout(this);
        ctxRow.setOrientation(LinearLayout.VERTICAL);
        // LiteRT GPU tests (fereko-pack-gpu)
        for (int w : new int[]{44, 76}) {
            if (!new File(new File(getFilesDir(), "pack"), "dec" + w + "_f32.tflite").isFile()) continue;
            for (boolean fp16 : new boolean[]{true, false}) {
                Button gb = button("GPU W" + w + (fp16 ? " 16bit" : " 32bit"), () -> gpuTest(w, fp16));
                gb.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                ctxRow.addView(gb);
            }
        }
        StringBuilder found = new StringBuilder();
        if (ctx != null) {
            java.util.Arrays.sort(ctx);
            for (File f : ctx) {
                String label = f.getName().replace("dacvae_decode.static44c2d.", "全体44 ").replace("dacvae_", "").replace("npu_", "").replace(".qnnctx.onnx", "");
                Button cb = button("ctx " + (label.isEmpty() ? "v79" : label), () -> generic(f.getName()));
                // vertical row: full width (the weight-based width of the horizontal rows would be 0 here)
                cb.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
                ctxRow.addView(cb);
                found.append(f.getName()).append(' ');
            }
        }
        // Reads this process's own logcat (allowed for an app's own logs) on a fresh thread, so it
        // works even while the worker is stuck in a native QNN call.
        Button logs = new Button(this);
        logs.setText("ログ");
        logs.setAllCaps(false);
        logs.setOnClickListener(v -> new Thread(this::dumpLog, "logcat").start());
        logs.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        Button kill = new Button(this);
        kill.setText("強制終了");
        kill.setAllCaps(false);
        kill.setOnClickListener(v -> {
            step("killed_by_user", null);
            Process.killProcess(Process.myPid());
        });
        kill.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        LinearLayout row2 = new LinearLayout(this);
        row2.addView(logs);
        row2.addView(kill);
        root.addView(row);
        root.addView(row2);

        status = new TextView(this);
        status.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(status);
        log = new TextView(this);
        log.setTypeface(Typeface.MONOSPACE);
        log.setTextSize(12);
        log.setTextIsSelectable(true);
        scroll = new ScrollView(this);
        // context buttons and the log scroll together, so the status line above stays visible
        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.addView(ctxRow);
        body.addView(log);
        scroll.addView(body);
        root.addView(scroll, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        root.setOnApplyWindowInsetsListener((v, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars());
            v.setPadding(pad + bars.left, pad + bars.top, pad + bars.right, pad + bars.bottom);
            return WindowInsets.CONSUMED;
        });
        setContentView(root);
        // verbose ORT (and QNN) logging for this process; read back with the ログ button
        try {
            ai.onnxruntime.OrtEnvironment.getEnvironment(ai.onnxruntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, "npu");
        } catch (Exception ignored) { }
        append("NPU 試験（別プロセス）。固まったら「ログ」でログをコピーしてから「強制終了」してね。");
        append("変換結果: " + (found.length() == 0 ? "なし" : found.toString()));
        Engines.cacheDir = getFilesDir();
        // Started from adb (see the NpuAuto alias in the manifest): run the listed steps, then exit.
        String auto = getIntent().getStringExtra("auto");
        if (auto != null) worker.execute(() -> runAuto(auto));
        // elapsed-time ticker, so a hang is visible as a growing number
        ui.postDelayed(new Runnable() {
            @Override public void run() {
                Object tag = status.getTag();
                if (t0 != 0 && !"done".equals(tag) && !"error".equals(tag)) status.setText(String.format(Locale.US, "%s  %.0f 秒", status.getTag() == null ? "" : status.getTag(), (System.nanoTime() - t0) / 1e9));
                ui.postDelayed(this, 1000);
            }
        }, 1000);
    }

    private Button button(String text, Runnable r) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setOnClickListener(v -> worker.execute(r));
        b.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        return b;
    }

    private void append(String s) {
        ui.post(() -> {
            log.append(s + "\n");
            scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
        });
    }

    /** Records a step before doing it (and shows it), so a native hang leaves a trace. */
    private void step(String name, String detail) {
        t0 = System.nanoTime();
        ui.post(() -> status.setTag(name));
        append(new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + " " + name + (detail == null ? "" : " " + detail));
        try {
            JSONObject o = new JSONObject();
            o.put("kind", "npu_step");
            o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
            o.put("app_version", MainActivity.APP_VERSION);
            o.put("step", name);
            if (detail != null) o.put("detail", detail);
            record(o);
        } catch (Exception ignored) { }
    }

    private void record(JSONObject o) throws IOException {
        try (FileWriter w = new FileWriter(new File(getExternalFilesDir(null), "runs.jsonl"), StandardCharsets.UTF_8, true)) {
            w.write(o.toString());
            w.write("\n");
        }
    }

    private void generic(String ctxName) {
        File pack = new File(getFilesDir(), "pack");
        try {
            JSONObject r = NpuTest.ctxGeneric(pack, ctxName, getApplicationInfo().nativeLibraryDir, this::append, this::step);
            JSONObject o = new JSONObject();
            o.put("kind", "npu");
            o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
            o.put("app_version", MainActivity.APP_VERSION);
            for (java.util.Iterator<String> it = r.keys(); it.hasNext(); ) {
                String k = it.next();
                o.put(k, r.get(k));
            }
            record(o);
            step("done", ctxName);
        } catch (Throwable t) {
            step("error", ctxName + ": " + t);
        }
    }

    private void gpuTest(int window, boolean fp16) {
        File pack = new File(getFilesDir(), "pack");
        try {
            JSONObject r = GpuTest.run(pack, window, fp16, this::append, this::step);
            JSONObject o = new JSONObject();
            o.put("kind", "npu");
            o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
            o.put("app_version", MainActivity.APP_VERSION);
            for (java.util.Iterator<String> it = r.keys(); it.hasNext(); ) {
                String k = it.next();
                o.put(k, r.get(k));
            }
            record(o);
            step("done", r.optString("backend"));
        } catch (Throwable t) {
            step("error", "gpu W" + window + (fp16 ? " 16bit" : " 32bit") + ": " + t);
        }
    }

    /**
     * Comma-separated steps, e.g. "gpuprep:16,synth:gpu16:int8dyn,synth::int8dyn".
     *   gpuprep:16|32          create the GPU decoder twice (first may compile, second should hit the cache)
     *   synth:<dec>:<dit>[:<speed>]  synthesize the reference line with that decoder/DiT (and speaking rate), save synth_<dec>….wav
     *   asrfile:<name>         our own recognizer (LocalAsr) on a 48 kHz WAV or raw 16 kHz .f32 file
     */
    private void runAuto(String auto) {
        step("auto_start", auto);
        for (String a : auto.split(",")) {
            String[] p = a.trim().split(":", -1);
            try {
                if (p[0].equals("gpuprep")) gpuPrep(p[1].equals("16"));
                else if (p[0].equals("synth")) synthBench(p.length > 1 ? p[1] : "", p.length > 2 ? p[2] : "",
                        p.length > 3 ? Float.parseFloat(p[3]) : 1f);
                else if (p[0].equals("sttfile")) sttFile(p[1], p.length > 2 ? Integer.parseInt(p[2]) : 16000);
                else if (p[0].equals("sttplay")) sttPlay(p[1]);
                else if (p[0].equals("asrfile")) asrFile(p[1]);
                else step("error", "unknown auto step " + a);
            } catch (Throwable t) {
                step("error", a + ": " + t);
            }
        }
        step("auto_done", auto);
        Engines.close();
        ui.post(this::finish);
        ui.postDelayed(() -> Process.killProcess(Process.myPid()), 500);
    }

    private void gpuPrep(boolean fp16) throws Exception {
        File pack = new File(getFilesDir(), "pack");
        for (int i = 0; i < 2; i++) {
            step("gpu_prep", (fp16 ? "16bit" : "32bit") + " #" + i);
            try (GpuWindowDecoder d = new GpuWindowDecoder(pack, getFilesDir(), GpuWindowDecoder.availableWindows(pack), fp16)) {
                JSONObject o = new JSONObject();
                o.put("kind", "npu");
                o.put("app_version", MainActivity.APP_VERSION);
                o.put("backend", "gpuprep-" + (fp16 ? "16bit" : "32bit"));
                o.put("create_ms", d.createMs);
                o.put("attempt", i);
                record(o);
                step("gpu_prep_done", (fp16 ? "16bit" : "32bit") + " #" + i + " " + d.createMs + " ms");
            }
        }
    }

    private static final String BENCH_TEXT = "…別に、キミのこと待ってたわけじゃないけど。まあ、来たならちょっとくらい話してあげてもいいよ。";
    private static float[] lastCpuAudio;

    private void synthBench(String dec, String dit, float speed) throws Exception {
        File pack = new File(getFilesDir(), "pack");
        step("synth_load", "dec=" + (dec.isEmpty() ? "cpu" : dec) + " dit=" + (dit.isEmpty() ? "fp32" : dit));
        long t0 = System.nanoTime();
        local.fereko.probe.core.IrodoriEngine eng = Engines.get(pack, 6, local.fereko.probe.core.IrodoriEngine.Accel.NONE, dit, dec);
        eng.speed = speed;
        long loadMs = (System.nanoTime() - t0) / 1_000_000;
        String caption;
        try (java.io.InputStream in = getAssets().open("voice.txt")) {
            caption = new String(in.readAllBytes(), StandardCharsets.UTF_8).trim();
        }
        local.fereko.probe.core.IrodoriEngine.Result r = null;
        long best = Long.MAX_VALUE;
        for (int i = 0; i < 3; i++) {   // first run includes warm-up
            step("synth_run", "#" + i);
            long t = System.nanoTime();
            r = eng.synthesize(BENCH_TEXT, null, 4, null, null, caption);
            long ms = (System.nanoTime() - t) / 1_000_000;
            if (i > 0) best = Math.min(best, ms);
        }
        double audioMs = r.audio.length * 1000.0 / local.fereko.probe.core.IrodoriEngine.SAMPLE_RATE;
        String name = "synth_" + (dec.isEmpty() ? "cpu" : dec) + (dit.isEmpty() ? "" : "_" + dit)
                + (speed == 1f ? "" : String.format(Locale.US, "_x%.2f", speed)) + ".wav";
        writeWav(new File(getExternalFilesDir(null), name), r.audio);
        JSONObject o = new JSONObject();
        o.put("kind", "npu");
        o.put("app_version", MainActivity.APP_VERSION);
        o.put("backend", "synth-" + (dec.isEmpty() ? "cpu" : dec) + "-" + (dit.isEmpty() ? "fp32" : dit));
        o.put("load_ms", loadMs);
        o.put("synth_ms", best);
        o.put("audio_ms", audioMs);
        o.put("rtf", best / audioMs);
        java.util.Map<String, Long> st = r.stageNanos;
        long dit4 = 0;
        for (int k = 0; k < 4; k++) dit4 += st.getOrDefault("dit_step" + k, 0L);
        o.put("dit_ms", dit4 / 1_000_000);
        o.put("decode_ms", st.getOrDefault("dacvae_decode", 0L) / 1_000_000);
        o.put("decode_first_ms", st.getOrDefault("decode_first_chunk", 0L) / 1_000_000);
        if (dec.isEmpty()) lastCpuAudio = r.audio.clone();
        else if (lastCpuAudio != null && lastCpuAudio.length == r.audio.length) {
            double sig = 0, err = 0;
            for (int i = 0; i < r.audio.length; i++) {
                double d = lastCpuAudio[i] - r.audio[i];
                sig += lastCpuAudio[i] * (double) lastCpuAudio[i];
                err += d * d;
            }
            o.put("snr_vs_cpu_db", 10 * Math.log10(sig / err));
        }
        o.put("wav", name);
        record(o);
        step("synth_done", o.toString());
    }

    /**
     * A 48 kHz PCM16 WAV from the external files dir: as is (rate 48000), or at 16 kHz through a
     * short windowed-sinc low-pass (cutoff 7 kHz) then every 3rd sample.
     */
    private short[] readWav(String name, int rate) throws IOException {
        byte[] wav = java.nio.file.Files.readAllBytes(new File(getExternalFilesDir(null), name).toPath());
        ByteBuffer bb = ByteBuffer.wrap(wav, 44, wav.length - 44).order(ByteOrder.LITTLE_ENDIAN);
        int n = (wav.length - 44) / 2;
        short[] s48 = new short[n];
        for (int i = 0; i < n; i++) s48[i] = bb.getShort();
        if (rate == 48000) return s48;
        float[] h = new float[31];
        double sum = 0;
        for (int k = 0; k < h.length; k++) {
            double x = k - 15, fc = 7000.0 / 48000;
            double sinc = x == 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
            h[k] = (float) (sinc * (0.54 - 0.46 * Math.cos(2 * Math.PI * k / 30)));
            sum += h[k];
        }
        for (int k = 0; k < h.length; k++) h[k] /= sum;
        short[] out = new short[n / 3];
        for (int i = 0; i < out.length; i++) {
            double acc = 0;
            for (int k = 0; k < h.length; k++) {
                int j = i * 3 + k - 15;
                if (j >= 0 && j < n) acc += h[k] * s48[j];
            }
            out[i] = (short) Math.max(-32768, Math.min(32767, Math.round(acc)));
        }
        return out;
    }

    /** Our own recognizer on a file: load time, per-stage times over 3 runs, text, memory. */
    private void asrFile(String name) throws Exception {
        float[] pcm;
        if (name.endsWith(".f32")) {
            pcm = local.fereko.probe.core.IrodoriEngine.readF32(new File(getExternalFilesDir(null), name));
        } else {
            short[] s = readWav(name, 16000);
            pcm = new float[s.length];
            for (int i = 0; i < s.length; i++) pcm[i] = s[i] / 32768f;
        }
        File pack = new File(getFilesDir(), "pack");
        step("asr_load", name);
        long t0 = System.nanoTime();
        local.fereko.probe.core.ReazonAsr asr = LocalAsr.get(pack);
        long loadMs = (System.nanoTime() - t0) / 1_000_000;
        org.json.JSONArray runs = new org.json.JSONArray();
        local.fereko.probe.core.ReazonAsr.Result r = null;
        for (int i = 0; i < 3; i++) {
            step("asr_run", "#" + i);
            long t = System.nanoTime();
            r = asr.recognize(pcm, pcm.length);
            runs.put(String.format(Locale.US, "total %d feat %d enc %d search %d", (System.nanoTime() - t) / 1_000_000,
                    r.featNanos / 1_000_000, r.encNanos / 1_000_000, r.searchNanos / 1_000_000));
        }
        android.os.Debug.MemoryInfo mi = new android.os.Debug.MemoryInfo();
        android.os.Debug.getMemoryInfo(mi);
        JSONObject o = new JSONObject();
        o.put("kind", "npu");
        o.put("app_version", MainActivity.APP_VERSION);
        o.put("backend", "asr-reazon");
        o.put("file", name);
        o.put("audio_ms", pcm.length * 1000L / 16000);
        o.put("load_ms", loadMs);
        o.put("runs", runs);
        o.put("text", r.text);
        o.put("pss_mb", mi.getTotalPss() / 1024);
        o.put("thermal", Probe.thermal(this));
        record(o);
        step("asr_file_done", o.toString());
        LocalAsr.close();
    }

    /**
     * Speech recognition of a 48 kHz WAV (e.g. a synth_*.wav) fed through a pipe with
     * EXTRA_AUDIO_SOURCE, the same path barge-in used to try (0.8.2/0.8.3).
     */
    private void sttFile(String name, int rate) throws Exception {
        short[] pcm = readWav(name, rate);
        byte[] pcm16k = new byte[pcm.length * 2];
        for (int i = 0; i < pcm.length; i++) {
            pcm16k[2 * i] = (byte) pcm[i];
            pcm16k[2 * i + 1] = (byte) (pcm[i] >> 8);
        }
        int frameBytes = rate / 50 * 2;
        java.util.concurrent.CountDownLatch done = new java.util.concurrent.CountDownLatch(1);
        StringBuilder events = new StringBuilder();
        String[] result = {null};
        android.os.ParcelFileDescriptor[] p = android.os.ParcelFileDescriptor.createPipe();
        step("stt_file", name + " " + (pcm16k.length / 2 * 1000L / rate) + " ms @" + rate);
        boolean[] ok = {false};
        ui.post(() -> {
            VoiceInput vi = new VoiceInput(this, new VoiceInput.Callback() {
                @Override public void event(String e) { events.append(e).append("; "); }
                @Override public void state(String s) { }
                @Override public void partial(String s) { }
                @Override public void result(String text, long sttMs, String mode) { result[0] = text; done.countDown(); }
                @Override public void error(String s) { result[0] = "ERROR " + s; done.countDown(); }
            });
            ok[0] = vi.startWithAudio(p[0], rate);
            if (!ok[0]) done.countDown();
        });
        Thread.sleep(300);
        try (OutputStream os = new android.os.ParcelFileDescriptor.AutoCloseOutputStream(p[1])) {
            for (int off = 0; off < pcm16k.length && ok[0]; off += frameBytes) {   // real time, 20 ms frames
                os.write(pcm16k, off, Math.min(frameBytes, pcm16k.length - off));
                Thread.sleep(20);
            }
            byte[] silence = new byte[frameBytes];
            for (int i = 0; i < 50 && ok[0]; i++) { os.write(silence); Thread.sleep(20); }   // 1 s of quiet
        } catch (IOException e) {
            events.append("write: ").append(e.getMessage()).append("; ");
        }
        boolean finished = done.await(20, java.util.concurrent.TimeUnit.SECONDS);
        p[0].close();
        JSONObject o = new JSONObject();
        o.put("kind", "npu");
        o.put("app_version", MainActivity.APP_VERSION);
        o.put("backend", "sttfile@" + rate);
        o.put("supported", ok[0]);
        o.put("finished", finished);
        o.put("events", events.toString());
        o.put("text", result[0] == null ? JSONObject.NULL : result[0]);
        record(o);
        step("stt_file_done", o.toString());
    }

    /** Plays a WAV through the speaker while the recognizer listens with its own microphone. */
    private void sttPlay(String name) throws Exception {
        byte[] wav = java.nio.file.Files.readAllBytes(new File(getExternalFilesDir(null), name).toPath());
        ByteBuffer bb = ByteBuffer.wrap(wav, 44, wav.length - 44).order(ByteOrder.LITTLE_ENDIAN);
        float[] audio = new float[(wav.length - 44) / 2];
        for (int i = 0; i < audio.length; i++) audio[i] = bb.getShort() / 32768f;
        java.util.concurrent.CountDownLatch ready = new java.util.concurrent.CountDownLatch(1), done = new java.util.concurrent.CountDownLatch(1);
        StringBuilder events = new StringBuilder();
        String[] result = {null};
        step("stt_play", name);
        ui.post(() -> {
            VoiceInput vi = new VoiceInput(this, new VoiceInput.Callback() {
                @Override public void event(String e) { events.append(e).append("; "); if (e.equals("ready")) ready.countDown(); }
                @Override public void state(String s) { }
                @Override public void partial(String s) { }
                @Override public void result(String text, long sttMs, String mode) { result[0] = text; done.countDown(); }
                @Override public void error(String s) { result[0] = "ERROR " + s; done.countDown(); }
            });
            vi.start();
        });
        ready.await(5, java.util.concurrent.TimeUnit.SECONDS);
        Probe.Player pl = new Probe.Player(audio, 48000);
        pl.start(48000);
        pl.waitDone();
        boolean finished = done.await(15, java.util.concurrent.TimeUnit.SECONDS);
        JSONObject o = new JSONObject();
        o.put("kind", "npu");
        o.put("app_version", MainActivity.APP_VERSION);
        o.put("backend", "sttplay");
        o.put("finished", finished);
        o.put("events", events.toString());
        o.put("text", result[0] == null ? JSONObject.NULL : result[0]);
        record(o);
        step("stt_play_done", o.toString());
    }

    private static void writeWav(File f, float[] audio) throws IOException {
        int n = audio.length, sr = local.fereko.probe.core.IrodoriEngine.SAMPLE_RATE;
        ByteBuffer b = ByteBuffer.allocate(44 + n * 2).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + n * 2).put("WAVE".getBytes(StandardCharsets.US_ASCII));
        b.put("fmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16).putShort((short) 1).putShort((short) 1)
                .putInt(sr).putInt(sr * 2).putShort((short) 2).putShort((short) 16);
        b.put("data".getBytes(StandardCharsets.US_ASCII)).putInt(n * 2);
        for (float v : audio) b.putShort((short) Math.round(Math.max(-1f, Math.min(1f, v)) * 32767f));
        java.nio.file.Files.write(f.toPath(), b.array());
    }

    private void tinyTest(boolean ctx) {
        File pack = new File(getFilesDir(), "pack");
        try {
            JSONObject r = NpuTest.tiny(pack, ctx, getApplicationInfo().nativeLibraryDir, this::append, this::step);
            JSONObject o = new JSONObject();
            o.put("kind", "npu");
            o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
            o.put("app_version", MainActivity.APP_VERSION);
            for (java.util.Iterator<String> it = r.keys(); it.hasNext(); ) {
                String k = it.next();
                o.put(k, r.get(k));
            }
            record(o);
            step("done", r.optString("backend"));
        } catch (Throwable t) {
            step("error", (ctx ? "tiny-ctx" : "tiny-online") + ": " + t);
        }
    }

    /** Saves this process's logcat to npu-log.txt and copies the QNN / FastRPC / ORT lines to the clipboard. */
    private void dumpLog() {
        try {
            java.lang.Process p = Runtime.getRuntime().exec(new String[]{"logcat", "-d", "-v", "time", "--pid", String.valueOf(Process.myPid())});
            String all = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            File out = new File(getExternalFilesDir(null), "npu-log.txt");
            java.nio.file.Files.write(out.toPath(), all.getBytes(StandardCharsets.UTF_8));
            StringBuilder pick = new StringBuilder();
            String[] lines = all.split("\n");
            java.util.List<String> keep = new java.util.ArrayList<>();
            for (String l : lines) {
                String u = l.toLowerCase(Locale.US);
                if (u.contains("qnn") || u.contains("rpc") || u.contains("dsp") || u.contains("htp") || u.contains("onnx")
                        || u.contains(" e/") || u.contains(" w/") || u.contains("error") || u.contains("fail")) keep.add(l);
            }
            for (int i = Math.max(0, keep.size() - 120); i < keep.size(); i++) pick.append(keep.get(i)).append('\n');
            String text = "npu log (" + lines.length + " lines, " + keep.size() + " picked)\n" + pick;
            ui.post(() -> {
                android.content.ClipboardManager cm = getSystemService(android.content.ClipboardManager.class);
                cm.setPrimaryClip(android.content.ClipData.newPlainText("npu-log", text));
                append("ログをコピーした（" + keep.size() + " 行のうち最後の " + Math.min(120, keep.size()) + " 行）。全部は npu-log.txt");
            });
        } catch (Exception e) {
            append("ログを取れなかった: " + e);
        }
    }

    /** Which HTP generation is this? Plugin device metadata, SoC info and the DSP libraries on the device. */
    private void info() {
        StringBuilder b = new StringBuilder();
        b.append("SOC_MODEL=").append(android.os.Build.SOC_MODEL).append(" SOC_MANUFACTURER=").append(android.os.Build.SOC_MANUFACTURER)
                .append(" HARDWARE=").append(android.os.Build.HARDWARE).append(" BOARD=").append(android.os.Build.BOARD).append("; ");
        for (String f : new String[]{"/sys/devices/soc0/soc_id", "/sys/devices/soc0/machine", "/sys/devices/soc0/family"}) {
            try {
                b.append(f).append('=').append(new String(java.nio.file.Files.readAllBytes(new File(f).toPath())).trim()).append("; ");
            } catch (Exception e) {
                b.append(f).append("=?; ");
            }
        }
        step("soc", b.toString());
        StringBuilder libs = new StringBuilder();
        for (String dir : new String[]{"/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/vendor/dsp", "/odm/lib/rfsa/adsp",
                "/system/lib/rfsa/adsp", "/vendor/lib64"}) {
            String[] names = new File(dir).list();
            if (names == null) { libs.append(dir).append(":-; "); continue; }
            int shown = 0;
            libs.append(dir).append(':');
            for (String n : names) {
                String l = n.toLowerCase(Locale.US);
                if (l.contains("htp") || l.contains("skel") || l.matches(".*v7[0-9].*") || l.matches(".*v8[0-9].*")) {
                    if (shown++ < 25) libs.append(' ').append(n);
                }
            }
            libs.append(" (").append(shown).append("); ");
        }
        step("dsp_libs", libs.toString());
        try {
            NpuTest.qnnDevices(getApplicationInfo().nativeLibraryDir, this::append, this::step);
        } catch (Throwable t) {
            step("error", "devices: " + t);
        }
        step("done", "info");
    }

    private void test(String backend) {
        File pack = new File(getFilesDir(), "pack");
        try {
            JSONObject r = NpuTest.run(pack, backend, 44, 6, getApplicationInfo().nativeLibraryDir, this::append, this::step);
            JSONObject o = new JSONObject();
            o.put("kind", "npu");
            o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
            o.put("app_version", MainActivity.APP_VERSION);
            for (java.util.Iterator<String> it = r.keys(); it.hasNext(); ) {
                String k = it.next();
                o.put(k, r.get(k));
            }
            record(o);
            step("done", backend);
        } catch (Throwable t) {
            step("error", backend + ": " + t);
        }
    }
}
