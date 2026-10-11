package local.fereko.probe;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.os.SystemClock;
import android.text.InputType;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import local.fereko.probe.core.IrodoriEngine;
import local.fereko.probe.core.SelfTest;

/**
 * On-device Irodori TTS probe. One job at a time on a worker thread; every measurement is
 * appended to runs.jsonl (app-specific external storage) and can be shared as text.
 */
public class MainActivity extends Activity {
    private static final int REQ_TREE = 1;
    static final String APP_VERSION = "0.19.1";
    private static final String ORT_VERSION = "1.30.0 + QNN plugin 2.6.0";
    private static final String DEFAULT_TEXT = "こんにちは。今日はいい天気ですね。";
    /** Extra lead time before playback starts, so estimate errors don't run the buffer dry. */
    private static final long START_MARGIN_NS = 150_000_000L;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final List<Button> buttons = new ArrayList<>();

    private EditText textInput, threadsInput, seedInput;
    private CheckBox playCheck;
    private TextView status, log;
    private ScrollView logScroll;

    private IrodoriEngine engine;
    private int engineThreads;
    private IrodoriEngine.Accel accel = IrodoriEngine.Accel.NONE;
    private IrodoriEngine.Accel engineAccel;
    private Button accelButton;
    private volatile boolean cancel;
    private volatile Runnable stopAudio;
    private long uiReadyMs;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        buildUi();
        uiReadyMs = SystemClock.uptimeMillis() - Process.getStartUptimeMillis();
        append("起動から画面表示まで " + uiReadyMs + " ms");
        append(packDir().isDirectory() ? "モデルは取り込み済み。「読み込み」を押してね" : "まず「モデル取り込み」で fereko-pack フォルダを選んでね");
    }

    // ---------- UI ----------

    private void buildUi() {
        int pad = dp(12);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        textInput = new EditText(this);
        textInput.setText(DEFAULT_TEXT);
        textInput.setMinLines(2);
        root.addView(textInput, matchWrap());

        LinearLayout opts = row();
        opts.addView(label("スレッド"));
        threadsInput = numberField("6");
        opts.addView(threadsInput);
        opts.addView(label("seed"));
        seedInput = numberField("0");
        opts.addView(seedInput);
        accelButton = new Button(this);
        accelButton.setAllCaps(false);
        accelButton.setOnClickListener(v -> {
            accel = IrodoriEngine.Accel.values()[(accel.ordinal() + 1) % IrodoriEngine.Accel.values().length];
            accelButton.setText(accelLabel());
        });
        accelButton.setText(accelLabel());
        buttons.add(accelButton);
        opts.addView(accelButton);
        playCheck = new CheckBox(this);
        playCheck.setText("再生");
        playCheck.setChecked(true);
        opts.addView(playCheck);
        root.addView(opts);

        LinearLayout r1 = row();
        r1.addView(button("モデル取り込み", v -> pickPack()));
        r1.addView(button("読み込み", v -> job(this::loadEngine)));
        r1.addView(button("自己テスト", v -> job(this::selfTest)));
        // back to the chat underneath (it is the launcher screen now) instead of opening a second one
        r1.addView(button("会話へ", v -> startActivity(new Intent(this, ChatActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP))));
        root.addView(r1);

        LinearLayout r2 = row();
        r2.addView(button("生成して再生", v -> job(() -> runSeries("single", 1))));
        r2.addView(button("5回", v -> job(() -> runSeries("warm5", 5))));
        r2.addView(button("10回連続", v -> job(() -> runSeries("burst10", 10))));
        r2.addView(button("NPU試験", v -> startActivity(new Intent(this, NpuActivity.class))));
        root.addView(r2);

        LinearLayout r3 = row();
        Button stop = new Button(this);
        stop.setText("停止");
        stop.setOnClickListener(v -> {
            cancel = true;
            Runnable s = stopAudio;
            if (s != null) s.run();
            append("停止を受け付けた（推論はステップの区切りで止まる）");
        });
        r3.addView(stop, weight());
        r3.addView(button("要約をコピー", v -> copySummary()));
        r3.addView(button("全部を共有", v -> shareRuns()));
        r3.addView(button("記録を消す", v -> confirmClear()));
        root.addView(r3);

        status = new TextView(this);
        status.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(status, matchWrap());

        log = new TextView(this);
        log.setTypeface(Typeface.MONOSPACE);
        log.setTextSize(12);
        log.setTextIsSelectable(true);
        logScroll = new ScrollView(this);
        logScroll.addView(log);
        root.addView(logScroll, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        // targetSdk 35+ draws edge-to-edge: keep the controls clear of the status bar, nav bar and keyboard.
        root.setOnApplyWindowInsetsListener((v, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.ime());
            v.setPadding(pad + bars.left, pad + bars.top, pad + bars.right, pad + bars.bottom);
            return WindowInsets.CONSUMED;
        });
        setContentView(root);
    }

    private LinearLayout row() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.HORIZONTAL);
        return l;
    }

    private TextView label(String s) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setPadding(dp(4), 0, dp(4), 0);
        return t;
    }

    private EditText numberField(String v) {
        EditText e = new EditText(this);
        e.setInputType(InputType.TYPE_CLASS_NUMBER);
        e.setText(v);
        e.setMinEms(2);
        return e;
    }

    private Button button(String text, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setOnClickListener(l);
        b.setLayoutParams(weight());
        buttons.add(b);
        return b;
    }

    private LinearLayout.LayoutParams weight() {
        return new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    private void append(String s) {
        ui.post(() -> {
            log.append(s + "\n");
            logScroll.post(() -> logScroll.fullScroll(View.FOCUS_DOWN));
        });
    }

    private void setStatus(String s) {
        ui.post(() -> status.setText(s));
    }

    private interface Job { void run() throws Exception; }

    /** Runs one job at a time; buttons are disabled meanwhile. */
    private void job(Job j) {
        cancel = false;
        for (Button b : buttons) b.setEnabled(false);
        worker.execute(() -> {
            try {
                j.run();
            } catch (IrodoriEngine.CancelledException e) {
                append("止めた");
            } catch (Throwable t) {
                append("エラー: " + t);
            } finally {
                setStatus("");
                ui.post(() -> { for (Button b : buttons) b.setEnabled(true); });
            }
        });
    }

    // ---------- model import ----------

    private File packDir() {
        return new File(getFilesDir(), "pack");
    }

    private void pickPack() {
        startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQ_TREE);
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req != REQ_TREE || res != RESULT_OK || data == null || data.getData() == null) return;
        Uri tree = data.getData();
        job(() -> {
            closeEngine();
            append("取り込み開始（3GB ほど。ハッシュも確かめる）");
            PackImporter.Report r = new PackImporter(getContentResolver(), tree).importTo(packDir(),
                    (file, done, total) -> setStatus(String.format(Locale.US, "取り込み中 %s  %.2f / %.2f GB", file, done / 1e9, total / 1e9)));
            append(String.format(Locale.US, "取り込み完了: %d ファイル %.2f GB, %.1f 秒（コピー＋SHA-256）", r.files, r.bytes / 1e9, r.millis / 1000.0));
            JSONObject o = base("import");
            o.put("files", r.files);
            o.put("bytes", r.bytes);
            o.put("import_and_hash_ms", r.millis);
            record(o);
        });
    }

    // ---------- engine ----------

    private int threads() {
        try { return Math.max(1, Integer.parseInt(threadsInput.getText().toString().trim())); }
        catch (NumberFormatException e) { return 6; }
    }

    private String accelLabel() {
        switch (accel) {
            case DECODER: return "XNNPACK:dec";
            case ALL: return "XNNPACK:全部";
            default: return "XNNPACK:なし";
        }
    }

    private void closeEngine() {
        Engines.close();
        engine = null;
    }

    private void loadEngine() throws Exception {
        if (!new File(packDir(), "pack.json").isFile()) throw new IOException("モデルが未取り込み");
        closeEngine();
        int th = threads();
        append("読み込み中（スレッド " + th + "）…");
        setStatus("読み込み中");
        JSONObject o = base("load");
        o.put("threads", th);
        o.put("accel", accel.name());
        o.put("before", Probe.thermal(this));
        long t0 = SystemClock.elapsedRealtime();
        try (Probe.MemorySampler mem = new Probe.MemorySampler(250)) {
            engine = Engines.get(packDir(), th, accel, "");
            engineThreads = th;
            engineAccel = accel;
            o.put("load_ms", SystemClock.elapsedRealtime() - t0);
            o.put("load_stage_ms", ms(engine.loadNanos));
            o.put("pss_peak_kb", mem.peakPssKb);
            o.put("native_heap_peak_kb", mem.peakNativeKb);
        }
        o.put("process_start_to_ready_ms", SystemClock.uptimeMillis() - Process.getStartUptimeMillis());
        o.put("ui_ready_ms", uiReadyMs);
        record(o);
        append("読み込み完了 " + o.getLong("load_ms") + " ms  " + o.get("load_stage_ms") + "  PSS峰 " + o.getLong("pss_peak_kb") / 1024 + " MB");
    }

    private void ensureEngine() throws Exception {
        if (engine == null || engineThreads != threads() || engineAccel != accel || !Engines.isLoaded(engineThreads, engineAccel, "")) loadEngine();
    }

    /** QNN feasibility: static-shape decoder on the NPU (HTP) and on the GPU, vs the CPU. */
    private void npuTest() throws Exception {
        closeEngine();   // free the CPU engine's memory first
        // htp-ctx: QNN context compiled on PC (fereko-pack-npu2); htp-c2d: 2-D conv model compiled here.
        for (String backend : new String[]{"htp-ctx", "htp-c2d"}) {
            for (int w : new int[]{44}) {
                setStatus("NPU試験 " + backend + " W" + w);
                JSONObject o = base("npu");
                try {
                    JSONObject r = NpuTest.run(packDir(), backend, w, threads(), getApplicationInfo().nativeLibraryDir, this::append);
                    for (java.util.Iterator<String> it = r.keys(); it.hasNext(); ) {
                        String k = it.next();
                        o.put(k, r.get(k));
                    }
                } catch (Throwable t) {
                    o.put("backend", backend);
                    o.put("window", w);
                    o.put("error", String.valueOf(t));
                    append("NPU試験 " + backend + " W" + w + " 失敗: " + t);
                }
                record(o);
            }
        }
    }

    private void selfTest() throws Exception {
        ensureEngine();
        setStatus("自己テスト中");
        String tok = SelfTest.tokens(engine.tokenizer(), packDir());
        append(tok);
        SelfTest.Parity p = SelfTest.parity(engine, packDir(), listener());
        append("PC との一致 " + (p.pass ? "OK " : "NG ") + p.detail);
        JSONObject o = base("selftest");
        o.put("tokens", tok);
        o.put("parity_pass", p.pass);
        o.put("parity", p.detail);
        record(o);
    }

    private IrodoriEngine.Listener listener() {
        return new IrodoriEngine.Listener() {
            @Override public void stage(String name, long nanos) { setStatus(name + " " + nanos / 1_000_000 + " ms"); }
            @Override public boolean cancelled() { return cancel; }
        };
    }

    private void runSeries(String series, int count) throws Exception {
        ensureEngine();
        String text = textInput.getText().toString();
        long seed;
        try { seed = Long.parseLong(seedInput.getText().toString().trim()); } catch (NumberFormatException e) { seed = 0; }
        boolean play = playCheck.isChecked();
        for (int i = 0; i < count && !cancel; i++) {
            JSONObject o = base("run");
            o.put("series", series);
            o.put("index", i);
            o.put("threads", engineThreads);
            o.put("accel", engineAccel.name());
            o.put("text", text);
            o.put("seed", seed);
            o.put("before", Probe.thermal(this));
            long baseline = android.os.Debug.getPss();
            final Probe.StreamPlayer[] sp = {null};
            final long[] windowFrames = {0};
            IrodoriEngine.AudioSink sink = !play ? null : (samples, idx, plan, decodeNs) -> {
                if (sp[0] == null) {
                    sp[0] = new Probe.StreamPlayer(plan.get(plan.size() - 1)[1] * IrodoriEngine.HOP, IrodoriEngine.SAMPLE_RATE);
                    Probe.StreamPlayer p = sp[0];
                    stopAudio = p::stop;
                }
                sp[0].write(samples);
                int[] c = plan.get(idx);
                windowFrames[0] += c[3] - c[2];
                long now = System.nanoTime();
                sp[0].startAt(idx == plan.size() - 1 ? now
                        : Probe.earliestStart(now, plan, idx, decodeNs / (double) windowFrames[0],
                                IrodoriEngine.HOP, IrodoriEngine.SAMPLE_RATE, START_MARGIN_NS));
            };
            long t0 = System.nanoTime();
            IrodoriEngine.Result r;
            long synthNs;
            try (Probe.MemorySampler mem = new Probe.MemorySampler(250)) {
                r = engine.synthesize(text, null, seed, listener(), sink);
                synthNs = System.nanoTime() - t0;
                o.put("pss_baseline_kb", baseline);
                o.put("pss_peak_kb", mem.peakPssKb);
                o.put("native_heap_peak_kb", mem.peakNativeKb);
                o.put("pss_interval_ms", mem.intervalMs);
            } catch (RuntimeException | ai.onnxruntime.OrtException e) {
                if (sp[0] != null) { sp[0].stop(); sp[0].finishAndWait(); }
                throw e;
            } finally {
                stopAudio = null;
            }
            double audioSec = r.audio.length / (double) IrodoriEngine.SAMPLE_RATE;
            o.put("normalized", r.normalized);
            o.put("tokens", r.tokenIds.length);
            o.put("pred_frames", r.predFrames);
            o.put("latent_frames", r.latentFrames);
            o.put("flatten_frame", r.flattenFrame);
            o.put("synth_ms", synthNs / 1e6);
            o.put("audio_ms", audioSec * 1000);
            o.put("rtf", synthNs / 1e9 / audioSec);
            o.put("stage_ms", ms(r.stageNanos));
            o.put("decode", "chunked ctx 6, plan 8/16/32/64, start margin " + START_MARGIN_NS / 1_000_000 + " ms");
            if (sp[0] != null) {
                Probe.StreamPlayer p = sp[0];
                stopAudio = p::stop;
                p.finishAndWait();
                stopAudio = null;
                o.put("button_to_play_call_ms", p.started ? (p.playCallNanos - t0) / 1e6 : JSONObject.NULL);
                if (p.firstFrameNanos > 0) {
                    o.put("button_to_first_frame_ms", (p.firstFrameNanos - t0) / 1e6);
                    o.put("first_frame_method", "AudioTrack.getTimestamp extrapolated to frame 0 (presentation, not acoustic)");
                } else {
                    o.put("button_to_first_frame_ms", JSONObject.NULL);
                    o.put("first_frame_method", "no timestamp within 2 s");
                }
                o.put("underruns", p.underruns);
            }
            o.put("after", Probe.thermal(this));
            record(o);
            append(summaryLine(o));
        }
    }

    // ---------- results ----------

    private JSONObject base(String kind) throws Exception {
        JSONObject o = new JSONObject();
        o.put("kind", kind);
        o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
        o.put("app_version", APP_VERSION);
        o.put("onnxruntime", ORT_VERSION + " CPU");
        o.put("model", "raratu/Onsei-Irodori-v4.1-Small-MF@a7542a4 (ios-mf-v2) FP32, no-ref, empty caption, 4 steps");
        o.put("device", Probe.device(this));
        return o;
    }

    private static JSONObject ms(Map<String, Long> m) throws Exception {
        JSONObject o = new JSONObject();
        for (Map.Entry<String, Long> e : m.entrySet()) o.put(e.getKey(), Math.round(e.getValue() / 1e5) / 10.0);
        return o;
    }

    /** One short line per record, small enough to paste from the phone. */
    static String summaryLine(JSONObject o) {
        String kind = o.optString("kind");
        StringBuilder b = new StringBuilder(o.optString("app_version")).append(' ');
        JSONObject after = o.optJSONObject("after");
        switch (kind) {
            case "run": {
                JSONObject st = o.optJSONObject("stage_ms");
                double dit = 0;
                if (st != null) for (int k = 0; k < 4; k++) dit += st.optDouble("dit_step" + k, 0);
                String ac = o.optString("accel", "NONE");
                if (!ac.equals("NONE")) b.append(ac.equals("DECODER") ? "XNNdec " : "XNNall ");
                b.append(String.format(Locale.US, "t%d %s#%d 生成%.0f 音声%.0f RTF%.2f 最初の音%s DiT%.0f dec%.0f(初%.0f) 途切れ%s PSS%.2fG 熱%d 電池%.0f℃",
                        o.optInt("threads"), o.optString("series"), o.optInt("index"),
                        o.optDouble("synth_ms"), o.optDouble("audio_ms"), o.optDouble("rtf"),
                        o.isNull("button_to_first_frame_ms") ? "-" : String.format(Locale.US, "%.0f", o.optDouble("button_to_first_frame_ms")),
                        dit, st == null ? 0 : st.optDouble("dacvae_decode", 0), st == null ? 0 : st.optDouble("decode_first_chunk", 0),
                        o.has("underruns") ? String.valueOf(o.optInt("underruns")) : "-",
                        o.optLong("pss_peak_kb") / 1048576.0,
                        after == null ? -1 : after.optInt("thermal_status"),
                        after == null ? 0 : after.optDouble("battery_temp_c")));
                break;
            }
            case "chat": {
                org.json.JSONArray sm = o.optJSONArray("synth_ms"), am = o.optJSONArray("audio_ms");
                double synth = 0, audio = 0;
                for (int k = 0; sm != null && am != null && k < sm.length() && k < am.length(); k++) {
                    synth += sm.optDouble(k);
                    audio += am.optDouble(k);
                }
                String dec = o.optString("dec", "fp32");
                if (o.optBoolean("reference")) b.append("登録声 ");
                b.append(String.format(Locale.US, "chat %s%s%s RTF%s 1文目合成%s 文字%s 文%s 音%s 計%d 文数%d 途切れ%d",
                        o.optString("dit"), dec.equals("fp32") ? "" : "/dec" + dec, o.optBoolean("caption") ? "+声説明" : "",
                        audio > 0 ? String.format(Locale.US, "%.2f", synth / audio) : "-",
                        sm != null && sm.length() > 0 ? String.valueOf(sm.optLong(0)) : "-",
                        o.isNull("send_to_first_text_ms") ? "-" : String.valueOf(o.optLong("send_to_first_text_ms")),
                        o.isNull("send_to_first_sentence_ms") ? "-" : String.valueOf(o.optLong("send_to_first_sentence_ms")),
                        o.isNull("send_to_first_sound_ms") ? "-" : String.valueOf(o.optLong("send_to_first_sound_ms")),
                        o.optLong("total_ms"), o.optInt("sentences"), o.optInt("underruns")));
                if (o.optLong("engine_load_ms") > 1000) b.append(" (声の読込").append(o.optLong("engine_load_ms")).append(")");
                if (o.has("stt_mode")) b.append(" 音声入力(").append(o.optString("stt_mode")).append(" 認識").append(o.optLong("stt_ms")).append("ms)");
                if (o.has("barge_in")) b.append(o.optBoolean("barge_in") ? " 割り込みあり" : "")
                        .append(String.format(Locale.US, " (echo%.0f thr%.0f max%.0f dB%s)", o.optDouble("barge_echo_db"),
                                o.optDouble("barge_threshold_db"), o.optDouble("barge_max_db"), o.optBoolean("barge_aec") ? " AEC" : ""));
                break;
            }
            case "npu_step":
                b.append("npu> ").append(o.optString("step"));
                if (o.has("detail")) b.append(" ").append(o.optString("detail"));
                break;
            case "npu":
                if (o.has("error")) {
                    b.append("npu ").append(o.optString("backend")).append(" W").append(o.optInt("window"))
                            .append(" 失敗 ").append(o.optString("error"));
                } else {
                    b.append(o.optBoolean("all_on_qnn") ? "全部QNN " : "一部CPU ");
                    if (o.optString("backend").startsWith("synth-")) {
                        b.append(String.format(Locale.US, "%s 生成%dms 音声%.0fms RTF%.2f DiT%dms dec%dms(初%dms)%s",
                                o.optString("backend"), o.optLong("synth_ms"), o.optDouble("audio_ms"), o.optDouble("rtf"),
                                o.optLong("dit_ms"), o.optLong("decode_ms"), o.optLong("decode_first_ms"),
                                o.has("snr_vs_cpu_db") ? String.format(Locale.US, " CPUとの差%.1fdB", o.optDouble("snr_vs_cpu_db")) : ""));
                        break;
                    }
                    if (o.optString("backend").startsWith("gpuprep")) {
                        b.append(o.optString("backend")).append(" 作成").append(o.optLong("create_ms")).append("ms #").append(o.optInt("attempt"));
                        break;
                    }
                    if (o.optString("backend").startsWith("gpu")) {
                        b.append(String.format(Locale.US, "%s 作成%dms 初回%dms 平均%dms RTF%.2f CPU%dms SNR%.1fdB",
                                o.optString("backend"), o.optLong("create_ms"), o.optLong("first_ms"), o.optLong("ms"),
                                o.optDouble("rtf_new_frames"), o.optLong("cpu_ms"), o.optDouble("snr_db")));
                        break;
                    }
                    if (o.optString("backend").startsWith("ctx:")) {
                        b.append(String.format(Locale.US, "npu %s 入力%s 作成%dms 初回%dms 平均%dms CPU%s SNR%s",
                                o.optString("backend"), o.optString("input_shape"), o.optLong("create_ms"), o.optLong("first_ms"), o.optLong("ms"),
                                o.has("cpu_ms") ? o.optLong("cpu_ms") + "ms" : "-", o.has("snr_db") ? String.format(Locale.US, "%.1fdB", o.optDouble("snr_db")) : "-"));
                        break;
                    }
                    b.append(String.format(Locale.US, "npu %s W%d 作成%dms 初回%dms 1回%dms CPU%dms RTF%.2f(CPU%.2f) SNR%.1fdB",
                            o.optString("backend"), o.optInt("window"), o.optLong("create_ms"), o.optLong("first_ms"),
                            o.optLong("ms"), o.optLong("cpu_ms"), o.optDouble("rtf_new_frames"), o.optDouble("cpu_rtf_new_frames"),
                            o.optDouble("snr_db")));
                }
                break;
            case "load":
                b.append(String.format(Locale.US, "load %s t%d %dms PSS%.2fG", o.optString("accel", "NONE"), o.optInt("threads"), o.optLong("load_ms"), o.optLong("pss_peak_kb") / 1048576.0));
                break;
            case "selftest":
                b.append("selftest ").append(o.optString("tokens")).append(o.optBoolean("parity_pass") ? " 一致OK " : " 一致NG ").append(o.optString("parity"));
                break;
            default:
                b.append(kind);
        }
        return b.toString();
    }

    private void copySummary() {
        StringBuilder b = new StringBuilder();
        JSONObject dev = null;
        for (String line : runsText().split("\n")) {
            if (line.isEmpty()) continue;
            try {
                JSONObject o = new JSONObject(line);
                if (dev == null) dev = o.optJSONObject("device");
                b.append(summaryLine(o)).append('\n');
            } catch (Exception ignored) { }
        }
        if (dev != null) b.insert(0, dev.optString("model") + " " + dev.optString("soc") + " Android " + dev.optString("android")
                + " RAM " + dev.optLong("ram_total_bytes") / 1_000_000_000 + "GB\n");
        ClipboardManager cm = getSystemService(ClipboardManager.class);
        cm.setPrimaryClip(ClipData.newPlainText("summary", b.toString()));
        append("要約をコピーした（" + b.length() + " 文字）");
    }

    private void confirmClear() {
        new android.app.AlertDialog.Builder(this)
                .setMessage("計測の記録（runs.jsonl）を消す？モデルは消えないよ")
                .setPositiveButton("消す", (d, w) -> {
                    //noinspection ResultOfMethodCallIgnored
                    runsFile().delete();
                    append("記録を消した");
                })
                .setNegativeButton("やめる", null)
                .show();
    }

    private File runsFile() {
        return new File(getExternalFilesDir(null), "runs.jsonl");
    }

    private synchronized void record(JSONObject o) throws IOException {
        try (FileWriter w = new FileWriter(runsFile(), StandardCharsets.UTF_8, true)) {
            w.write(o.toString());
            w.write("\n");
        }
    }

    private String runsText() {
        try {
            File f = runsFile();
            return f.isFile() ? new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8) : "";
        } catch (IOException e) {
            return "";
        }
    }

    private void shareRuns() {
        Intent i = new Intent(Intent.ACTION_SEND);
        i.setType("text/plain");
        i.putExtra(Intent.EXTRA_SUBJECT, "fereko TTS probe runs.jsonl");
        i.putExtra(Intent.EXTRA_TEXT, runsText());
        startActivity(Intent.createChooser(i, "結果を共有"));
    }

    @Override
    protected void onDestroy() {
        cancel = true;
        worker.shutdownNow();
        closeEngine();
        super.onDestroy();
    }
}
