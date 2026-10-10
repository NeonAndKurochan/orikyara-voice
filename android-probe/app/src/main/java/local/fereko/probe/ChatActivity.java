package local.fereko.probe;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
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
import local.fereko.probe.core.MiniJson;
import local.fereko.probe.core.RelayClient;
import local.fereko.probe.core.SentenceSplitter;
import local.fereko.probe.core.ChatMemory;
import local.fereko.probe.core.Affection;

/**
 * Talk to ふぇれ子: text in, Claude (through the relay) out, each sentence spoken on-device
 * while the rest of the reply is still streaming. The API key never touches the phone; only the
 * relay token (imported from relay-token.txt) does. History is append-only, kept as received.
 */
public class ChatActivity extends Activity {
    private static final int REQ_TOKEN = 2;
    private static final int REQ_MIC = 3;
    private static final int THREADS = 6;
    /** Pause in the reply stream after which a finished sentence is spoken without waiting for more. */
    private static final long IDLE_RELEASE_NS = 250_000_000L;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final List<Button> buttons = new ArrayList<>();

    private TextView log;
    private ScrollView logScroll;
    private EditText input, captionInput;
    private Button ditButton, decButton, refButton, asrButton, modelButton, speedButton;
    private MicButton micButton;
    private android.widget.CheckBox handsFree, bargeInBox;
    /** The previous reply was cut off by the user talking over it: tell Claude in the next message. */
    private volatile boolean interruptedLast;
    private VoiceInput voice;
    /** Our own recognizer's microphone while listening for a normal turn (🎤 / hands-free). */
    private volatile BargeInDetector ownMic;
    private SharedPreferences prefs;

    private String persona;
    /** Who the user is talking to (ふぇれ子 or a character folder): name, persona, voice, look, data. */
    private CharacterProfile chr;

    /** Preference key for a per-character value (ふぇれ子 keeps the original keys). */
    private String k(String key) { return chr.key(key); }
    /** What ふぇれ子 remembers from older turns (ChatMemory), in files/chat/memory.md. */
    private volatile String memo = "";
    private volatile boolean compacting;
    private List<Object> history;
    private volatile boolean cancel;
    private volatile RelayClient activeRelay;
    private volatile TtsPipeline activeTts;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        if (getActionBar() != null) getActionBar().hide();   // full-screen picture; the name is in our own top bar
        ui.postDelayed(() -> handleAutoIntent(getIntent()), 3000);   // after the voice preload has started
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        prefs = getSharedPreferences("chat", MODE_PRIVATE);
        Engines.cacheDir = getFilesDir();
        chr = CharacterProfile.load(this, prefs.getString("character", CharacterProfile.BUILTIN));
        persona = chr.persona;
        history = loadHistory();
        memo = readMemo();
        buildUi();
        for (Object o : history) show(MiniJson.obj(o));
        // no greeting when opened by an adb test (.ChatAuto): it would be a real Claude turn in the history
        boolean autoTest = getIntent() != null && getIntent().getComponent() != null
                && getIntent().getComponent().getClassName().endsWith(".ChatAuto");
        if (!tokenFile().isFile()) append("（まず「合言葉」で relay-token.txt を取り込んでね）");
        else if (b != null || autoTest || !scheduleGreeting()) warmCache();   // the greeting itself warms the cache
        if (!new File(packDir(), "pack.json").isFile()) {
            append("（声のモデルがまだ。⋯ の設定の「試験画面」で取り込んでね）");
            prepError = "モデルがまだ入ってない";
            prep = PREP_FAILED;
        }
        else preloadEngine();
        voice = new VoiceInput(this, new VoiceInput.Callback() {
            @Override public void event(String e) { logEvent("stt", e); }
            @Override public void state(String s) {
                setStatus(s);
                if (!s.startsWith("聞いてるよ") && !s.startsWith("認識中")) hint(s);   // e.g. the network-recognizer notice
            }
            @Override public void partial(String s) { input.setText(s); }
            @Override public void result(String text, long sttMs, String mode) {
                closeBargeFd();
                input.setText("");
                sendText(text, sttMs, mode);
            }
            @Override public void error(String s) {
                closeBargeFd();
                hint(s);
            }
        });
    }

    /**
     * Prepares Claude's prompt cache (persona + history) while the user is still getting ready to
     * talk, so the first reply is as quick as the later ones (cold: 2-5 s to the first text).
     * The relay keeps it for an hour.
     */
    private void warmCache() {
        List<Object> msgs = new ArrayList<>(history);
        // must end with a user turn: the previous turns' cache ends at the last user message
        while (!msgs.isEmpty() && "assistant".equals(MiniJson.obj(msgs.get(msgs.size() - 1)).get("role"))) msgs.remove(msgs.size() - 1);
        if (msgs.isEmpty()) {
            Map<String, Object> m = new java.util.LinkedHashMap<>();
            m.put("role", "user");
            m.put("content", "（準備）");
            msgs.add(m);
        }
        String model = model();
        new Thread(() -> {
            try {
                long t = System.nanoTime();
                Map<String, Object> u = relay().warm(systemPrompt(), msgs, model);
                logEvent("warm", model + " " + (System.nanoTime() - t) / 1_000_000 + " ms " + MiniJson.write(u));
            } catch (Exception e) {
                logEvent("warm", model + " failed: " + e.getMessage());
            }
        }, "cache-warm").start();
    }

    /** Loads the voice in the background when the screen opens, so the first reply doesn't wait ~5 s. */
    private void preloadEngine() {
        String dit = dit(), dec = dec();
        new Thread(() -> {
            try {
                long t = System.nanoTime();
                Engines.get(packDir(), THREADS, IrodoriEngine.Accel.NONE, dit, dec);
                double voiceSec = (System.nanoTime() - t) / 1e9;
                if (ownAsr()) {
                    t = System.nanoTime();
                    LocalAsr.get(packDir());
                    setStatus(String.format(Locale.US, "声の準備OK（%.1f 秒）、耳の準備OK（%.1f 秒）", voiceSec, (System.nanoTime() - t) / 1e9));
                } else {
                    setStatus(String.format(Locale.US, "声の準備OK（%.1f 秒）", voiceSec));
                }
                prep = PREP_READY;
            } catch (Exception e) {
                setStatus("声の準備に失敗: " + e.getMessage());
                prepError = e.getMessage();
                prep = PREP_FAILED;
                hint("声の準備に失敗: " + e.getMessage());
            }
        }, "engine-preload").start();
    }

    private void startListening() {
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.RECORD_AUDIO}, REQ_MIC);
            return;
        }
        if (ownAsr()) startOwnListening();
        else voice.start();
    }

    /** Our own recognizer is in the pack and selected (「耳」 button). */
    private boolean ownAsr() {
        return !"android".equals(prefs.getString("asr", "")) && LocalAsr.available(packDir());
    }

    /** 🎤 / hands-free with our own recognizer: listen until the user stops talking, then recognize. */
    private void startOwnListening() {
        setStatus("聞いてるよ…（" + LocalAsr.MODE + "）");
        logEvent("own_listen", "start");
        ownMic = new BargeInDetector(BargeInDetector.Mode.LISTEN, () -> true, new BargeInDetector.Listener() {
            @Override public void bargeIn() { setStatus("聞き取り中…"); }
            @Override public void utterance(float[] pcm, int n) {
                ownMic = null;
                recognizeAndSend(pcm, n, false);
            }
        });
    }

    /**
     * Runs our recognizer on a captured utterance (on the microphone thread) and sends the text.
     * @param afterBargeIn the user cut ふぇれ子 off; if nothing usable was heard, listen again
     */
    private void recognizeAndSend(float[] pcm, int n, boolean afterBargeIn) {
        if (n == 0) {
            ui.post(() -> {
                recognizing = false;
                hint("聞き取れなかった");
            });
            return;
        }
        setStatus("認識中…");
        recognizing = true;
        try {
            long t = System.nanoTime();
            local.fereko.probe.core.ReazonAsr.Result r = LocalAsr.get(packDir()).recognize(pcm, n);
            long ms = (System.nanoTime() - t) / 1_000_000;
            logEvent("own_asr", String.format(Locale.US, "audio %.1fs, %d ms (feat %d enc %d search %d), %d chars",
                    r.audioSec, ms, r.featNanos / 1_000_000, r.encNanos / 1_000_000, r.searchNanos / 1_000_000, r.text.length()));
            ui.post(() -> {
                if (r.text.isEmpty()) {
                    recognizing = false;
                    hint("聞き取れなかった");
                    if (afterBargeIn) startListening();
                } else {
                    sendText(r.text, ms, LocalAsr.MODE);
                }
            });
        } catch (Exception e) {
            ui.post(() -> {
                recognizing = false;
                hint("聞き取りでエラー: " + e.getMessage());
            });
        }
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] results) {
        super.onRequestPermissionsResult(req, perms, results);
        if (req == REQ_MIC && results.length > 0 && results[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) startListening();
        else if (req == REQ_MIC) hint("マイクが使えないと、声では話せないよ");
    }

    private android.os.ParcelFileDescriptor bargeReadFd;

    /**
     * Barge-in fired without our own recognizer: release the microphone, beep, and start the recognizer, so the user speaks
     * after the beep. (Feeding our own captured audio to the on-device recognizer via
     * EXTRA_AUDIO_SOURCE was tried in 0.8.2/0.8.3: it detects speech but returns empty text, and it
     * also ignores audio the phone plays itself, so the words that triggered the barge-in can't be
     * recognized with the platform recognizer.)
     */
    private void handOff(BargeInDetector barge) {
        new Thread(() -> {
            barge.abort();   // the recognizer needs the microphone
            try {
                android.media.ToneGenerator tg = new android.media.ToneGenerator(android.media.AudioManager.STREAM_MUSIC, 60);
                tg.startTone(android.media.ToneGenerator.TONE_PROP_BEEP, 120);
                Thread.sleep(160);
                tg.release();
            } catch (Exception ignored) { }
            logEvent("barge_handoff", "beep + listen");
            ui.post(() -> {
                hint("ピッのあとに話してね");
                startListening();
            });
        }, "barge-handoff").start();
    }

    /** One line in runs.jsonl for voice/barge-in diagnostics. */
    private void logEvent(String what, String detail) {
        try {
            JSONObject o = new JSONObject();
            o.put("kind", "voice_event");
            o.put("time", new SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(new Date()));
            o.put("app_version", MainActivity.APP_VERSION);
            o.put("what", what);
            o.put("detail", detail);
            record(o);
        } catch (Exception ignored) { }
    }

    private void closeBargeFd() {
        if (bargeReadFd != null) {
            try { bargeReadFd.close(); } catch (IOException ignored) { }
            bargeReadFd = null;
        }
    }

    private void sendText(String text, long sttMs, String sttMode) {
        ui.post(() -> {
            stickToBottom = true;
            logScroll.post(() -> logScroll.scrollTo(0, Math.max(0, log.getHeight() - logScroll.getHeight())));
        });
        prefs.edit().putString(k("caption"), captionInput.getText().toString().trim()).apply();
        job(() -> talk(text, sttMs, sttMode));
    }

    // ---------- UI ----------

    /**
     * Full-screen layout: a blurred background (room / outside), ふぇれ子 standing over it, and the
     * conversation on a dark glass band at the bottom that fades out upwards, so she stays visible
     * behind it. Monochrome like her outfit, with her red eye colour as the only accent. The old
     * setting buttons live in a sheet behind the ⋯ button.
     */
    private void buildUi() {
        android.widget.FrameLayout root = new android.widget.FrameLayout(this);
        root.setBackgroundColor(0xFF101014);
        android.widget.FrameLayout.LayoutParams full = new android.widget.FrameLayout.LayoutParams(-1, -1);

        // 1. background, blurred
        bgView = new android.widget.ImageView(this);
        bgView.setScaleType(android.widget.ImageView.ScaleType.CENTER_CROP);
        bgView.setRenderEffect(android.graphics.RenderEffect.createBlurEffect(dp(5), dp(5), android.graphics.Shader.TileMode.CLAMP));
        root.addView(bgView, full);
        View dim = new View(this);
        dim.setBackgroundColor(0x26000000);
        root.addView(dim, new android.widget.FrameLayout.LayoutParams(-1, -1));
        showBackground();

        // 2. ふぇれ子
        java.io.File vrm = chr.vrm(packDir());
        android.graphics.Bitmap stand = chr.stand(this);
        if (vrm != null && prefs.getBoolean("avatar3d", true)) {
            // 3D: transparent WebView over the background; the 2D picture is not drawn
            avatar = new AvatarView(this, chr.modelDir(packDir()), vrm, () -> avatarState, () -> {
                TtsPipeline t = activeTts;
                return t == null ? 0 : t.level();
            }, json -> {
                try { record(new JSONObject(json)); } catch (Exception ignored) { }
            }, (name, seconds) -> {
                // turning around shows her tail, which is behind the conversation: fade it meanwhile
                if (!name.equals("turn_around")) return;
                ui.post(() -> {
                    for (View v : new View[]{logScroll, bottomScrim}) v.animate().alpha(0.08f).setDuration(400).start();
                    ui.postDelayed(() -> {
                        for (View v : new View[]{logScroll, bottomScrim}) v.animate().alpha(1f).setDuration(400).start();
                    }, (long) (seconds * 1000) - 600);
                });
            });
            root.addView(avatar, new android.widget.FrameLayout.LayoutParams(-1, -1));
            prefer60Hz();
        } else if (stand != null) {
            root.addView(new StandView(this, stand, () -> {
                TtsPipeline t = activeTts;
                return t == null ? 0 : t.level();
            }, 10, 830, 0.095f, 0.93f, 540), new android.widget.FrameLayout.LayoutParams(-1, -1));   // ears to knees, like the 3D framing will be
        }

        // 3. scrims: top for the status bar icons, bottom for the conversation
        View topScrim = new View(this);
        topScrim.setBackground(new android.graphics.drawable.GradientDrawable(
                android.graphics.drawable.GradientDrawable.Orientation.TOP_BOTTOM, new int[]{0x99000000, 0x00000000}));
        root.addView(topScrim, new android.widget.FrameLayout.LayoutParams(-1, dp(120), android.view.Gravity.TOP));
        bottomScrim = new View(this);
        bottomScrim.setBackground(new android.graphics.drawable.GradientDrawable(
                android.graphics.drawable.GradientDrawable.Orientation.BOTTOM_TOP, new int[]{0xF0101014, 0xD0101014, 0x00101014}));
        android.widget.FrameLayout.LayoutParams bsl = new android.widget.FrameLayout.LayoutParams(-1, 0, android.view.Gravity.BOTTOM);
        root.addView(bottomScrim, bsl);
        root.addOnLayoutChangeListener((v, l, t, r, b, ol, ot, or, ob) -> {
            int want = (int) ((b - t) * 0.56f);
            if (bottomScrim.getLayoutParams().height != want) {
                bottomScrim.getLayoutParams().height = want;
                bottomScrim.post(bottomScrim::requestLayout);
            }
        });

        // 4. foreground: top bar, conversation, input
        LinearLayout fg = new LinearLayout(this);
        fg.setOrientation(LinearLayout.VERTICAL);
        root.addView(fg, new android.widget.FrameLayout.LayoutParams(-1, -1));

        LinearLayout bar = row();
        bar.setGravity(android.view.Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(16), dp(6), dp(8), dp(6));
        TextView name = new TextView(this);
        name.setText(chr.name);
        name.setTextColor(0xFFFFFFFF);
        name.setTextSize(20);
        name.setTypeface(Typeface.DEFAULT_BOLD);
        name.setShadowLayer(dp(4), 0, 0, 0x80000000);
        bar.addView(name);
        statePill = new TextView(this);
        statePill.setTextSize(12);
        statePill.setTextColor(0xFFFFFFFF);
        statePill.setPadding(dp(10), dp(3), dp(10), dp(3));
        statePill.setOnClickListener(v -> {
            if (prep == PREP_LOADING) hint("声と耳を用意してるとこ。もうちょっと待って");
            else if (prep == PREP_FAILED) hint("声の準備ができなかった: " + prepError);
        });
        LinearLayout.LayoutParams spl = new LinearLayout.LayoutParams(-2, -2);
        spl.leftMargin = dp(10);
        bar.addView(statePill, spl);
        bar.addView(new View(this), new LinearLayout.LayoutParams(0, 1, 1f));
        bgButton = chip(bgLabel(), v -> {
            prefs.edit().putString("bg", "outside".equals(prefs.getString("bg", "room")) ? "room" : "outside").apply();
            bgButton.setText(bgLabel());
            showBackground();
        });
        bar.addView(bgButton);
        TextView more = chip("⋯", v -> openSettings());
        more.setContentDescription("設定");
        LinearLayout.LayoutParams ml = new LinearLayout.LayoutParams(-2, -2);
        ml.leftMargin = dp(8);
        bar.addView(more, ml);
        fg.addView(bar);

        fg.addView(new View(this), new LinearLayout.LayoutParams(-1, 0, 1f));   // ふぇれ子 shows through here

        log = new TextView(this);
        log.setTextIsSelectable(true);
        log.setLineSpacing(dp(2), 1f);
        log.setShadowLayer(dp(3), 0, dp(1), 0xCC000000);
        log.setPadding(dp(20), dp(8), dp(20), dp(8));
        chatLog = new ChatLog(log);
        logScroll = new ScrollView(this);
        logScroll.setVerticalScrollBarEnabled(false);
        logScroll.setVerticalFadingEdgeEnabled(true);   // older lines fade out upwards
        logScroll.setFadingEdgeLength(dp(56));
        logScroll.addView(log);
        fg.addView(logScroll, new LinearLayout.LayoutParams(-1, 0, 0.62f));
        // stay on the newest line while the text grows or the panel changes size, unless the user
        // has scrolled up to read older lines
        logScroll.setOnScrollChangeListener((v, x, y, ox, oy) -> stickToBottom = !v.canScrollVertically(1));
        View.OnLayoutChangeListener stick = (v, l, t, r, b, ol, ot, or, ob) -> {
            if (stickToBottom) logScroll.post(() -> logScroll.scrollTo(0, Math.max(0, log.getHeight() - logScroll.getHeight())));
        };
        log.addOnLayoutChangeListener(stick);
        logScroll.addOnLayoutChangeListener(stick);

        // short messages for the user (「聞き取れなかった」 etc.): shown for a few seconds, then gone
        hintView = new TextView(this);
        hintView.setTextSize(13);
        hintView.setTextColor(0xFFFFFFFF);
        hintView.setPadding(dp(14), dp(6), dp(14), dp(6));
        android.graphics.drawable.GradientDrawable hb = new android.graphics.drawable.GradientDrawable();
        hb.setColor(0xB3202026);
        hb.setCornerRadius(dp(16));
        hintView.setBackground(hb);
        hintView.setVisibility(View.GONE);
        LinearLayout.LayoutParams hl = new LinearLayout.LayoutParams(-2, -2);
        hl.gravity = android.view.Gravity.CENTER_HORIZONTAL;
        hl.bottomMargin = dp(6);
        fg.addView(hintView, hl);

        LinearLayout inputBar = row();
        inputBar.setGravity(android.view.Gravity.CENTER_VERTICAL);
        inputBar.setPadding(dp(12), dp(4), dp(12), dp(10));
        micButton = new MicButton(this);
        micButton.setOnClickListener(v -> {
            BargeInDetector m = ownMic;
            if (m != null) m.finish();
            else if (voice.listening) voice.stop();
            else startListening();
        });
        inputBar.addView(micButton, new LinearLayout.LayoutParams(dp(54), dp(54)));
        LinearLayout pill = row();
        pill.setGravity(android.view.Gravity.CENTER_VERTICAL);
        android.graphics.drawable.GradientDrawable pillBg = new android.graphics.drawable.GradientDrawable();
        pillBg.setColor(0x59000000);
        pillBg.setStroke(dp(1), 0x40FFFFFF);
        pillBg.setCornerRadius(dp(24));
        pill.setBackground(pillBg);
        pill.setPadding(dp(16), 0, dp(6), 0);
        input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        input.setMaxLines(3);
        input.setBackground(null);
        input.setTextColor(0xFFFFFFFF);
        input.setHintTextColor(0x80FFFFFF);
        input.setHint("ふぇれ子に話しかける");
        input.setTextSize(15);
        pill.addView(input, new LinearLayout.LayoutParams(0, -2, 1f));
        sendButton = chip("送る", v -> {
            if (activeRelay != null || activeTts != null) {   // a turn is running: this button stops it
                cancel = true;
                RelayClient r = activeRelay;
                if (r != null) r.cancel();
                TtsPipeline t = activeTts;
                if (t != null) t.cancel();
                return;
            }
            String text = input.getText().toString().trim();
            if (text.isEmpty()) return;
            input.setText("");
            sendText(text, -1, null);
        });
        pill.addView(sendButton);
        LinearLayout.LayoutParams pl = new LinearLayout.LayoutParams(0, dp(48), 1f);
        pl.leftMargin = dp(10);
        inputBar.addView(pill, pl);
        fg.addView(inputBar);

        buildSettingsPanel();

        fg.setOnApplyWindowInsetsListener((v, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.ime());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return WindowInsets.CONSUMED;
        });
        setContentView(root);
        ui.post(stateTick);
    }

    private android.widget.ImageView bgView;
    /** Cues of the turn being spoken (one per said sentence) and the last one sent to the 3D page. */
    private volatile List<JSONObject> activeCues;
    private int lastCue = -1;

    /** The emotion emoji a sentence ends with (the persona puts at most one there), or null. */
    private static String emotionOf(String s) {
        for (String e : new String[]{"😒", "🙄", "😏", "😪", "🤭"}) if (s.contains(e)) return e;
        return null;
    }
    private View bottomScrim;

    /**
     * The 3D page draws at most 60 fps; on a 90/120 Hz screen that means uneven frame pacing and
     * wasted power, so ask for a 60 Hz display mode of the same resolution while the chat is shown.
     */
    private void prefer60Hz() {
        android.view.Display d = getDisplay();
        if (d == null) return;
        android.view.Display.Mode cur = d.getMode();
        for (android.view.Display.Mode m : d.getSupportedModes()) {
            if (Math.abs(m.getRefreshRate() - 60f) < 1f && m.getPhysicalWidth() == cur.getPhysicalWidth()
                    && m.getPhysicalHeight() == cur.getPhysicalHeight()) {
                WindowManager.LayoutParams lp = getWindow().getAttributes();
                lp.preferredDisplayModeId = m.getModeId();
                getWindow().setAttributes(lp);
                return;
            }
        }
    }

    /**
     * adb test hook (only through the .ChatAuto alias, which needs the DUMP permission): speak a
     * fixed line n times through the normal voice pipeline, without Claude and without touching the
     * history, so the 3D frame rate can be measured while the voice is being synthesized.
     *   am start -n local.fereko.probe/.ChatAuto --es say "テキスト" --ei times 3
     */
    private void handleAutoIntent(Intent in) {
        if (in == null || in.getComponent() == null || !in.getComponent().getClassName().endsWith(".ChatAuto")) return;
        String motion = in.getStringExtra("motion");   // --es motion turn_around
        if (motion != null && avatar != null) ui.postDelayed(() -> avatar.playReaction(motion), 4000);
        // demo video: --ez rec true keeps her voice (files/rec/turn_*.f32 + .json with the wall-clock start),
        // --ez reset true starts the conversation over, --ez greet true greets first,
        // --es script "line|line|…" sends those lines one after another as the user's messages
        if (in.getBooleanExtra("rec", false)) recordVoice = true;
        if (in.getBooleanExtra("reset", false)) job(() -> resetHistory());
        String script = in.getStringExtra("script");
        boolean greet = in.getBooleanExtra("greet", false);
        if (greet || script != null) {
            List<String> lines = new ArrayList<>();
            if (greet) lines.add(null);   // null = the app-opened event
            if (script != null) for (String l : script.split("\\|")) if (!l.trim().isEmpty()) lines.add(l.trim());
            runScript(lines, 0);
        }
        String text = in.getStringExtra("say");
        if (text == null || text.isEmpty()) return;
        int times = Math.max(1, in.getIntExtra("times", 1));
        String caption = prefs.getString(k("caption"), "");
        boolean useRef = prefs.getBoolean(k("use_ref"), true);
        String dit = dit(), dec = in.hasExtra("dec") ? in.getStringExtra("dec") : dec();   // --es dec "" = CPU, for comparisons
        float speed = speed();
        job(() -> {
            for (int i = 0; i < times && !cancel; i++) {
                long t0 = System.nanoTime();
                java.util.concurrent.FutureTask<IrodoriEngine> engineTask = new java.util.concurrent.FutureTask<>(() -> {
                    IrodoriEngine e = Engines.get(packDir(), THREADS, IrodoriEngine.Accel.NONE, dit, dec);
                    e.speed = speed;
                    e.setReference(chr.voiceDir(packDir()));
                    return e;
                });
                new Thread(engineTask, "engine-load").start();
                TtsPipeline tts = new TtsPipeline(engineTask, caption, 0, t0, useRef);
                activeTts = tts;
                // the same cues as a real reply: emoji -> emotion, a ［…］ tag -> motion of the sentence it
                // starts (tags are not spoken)
                StringBuilder visible = new StringBuilder();
                List<int[]> tagAt = new ArrayList<>();       // {visible index, tag number}
                List<String> tagNames = new ArrayList<>();
                java.util.regex.Matcher m = java.util.regex.Pattern.compile("［([^］]{1,20})］").matcher(text);
                int last = 0;
                while (m.find()) {
                    visible.append(text, last, m.start());
                    tagAt.add(new int[]{visible.length(), tagNames.size()});
                    tagNames.add(m.group(1));
                    last = m.end();
                }
                visible.append(text.substring(last));
                SentenceSplitter sp = new SentenceSplitter();
                List<String> sentences = new ArrayList<>(sp.feed(visible.toString()));
                sentences.addAll(sp.flush());
                List<JSONObject> cues = new ArrayList<>();
                int from = 0;
                for (int si = 0; si < sentences.size(); si++) {
                    int to = visible.indexOf(sentences.get(si), from) + sentences.get(si).length();
                    org.json.JSONArray acts = new org.json.JSONArray();
                    for (int[] ta : tagAt) if (ta[0] >= from && ta[0] < Math.max(to, from + 1)) acts.put(tagNames.get(ta[1]));
                    JSONObject c = new JSONObject();
                    c.put("text", sentences.get(si));
                    c.put("emotion", emotionOf(sentences.get(si)));
                    c.put("actions", acts);
                    c.put("first", si == 0);
                    cues.add(c);
                    from = Math.max(from, to);
                }
                activeCues = cues;
                for (String s : sentences) tts.say(s);
                tts.end();
                tts.await();
                activeTts = null;
                JSONObject o = new JSONObject();
                o.put("kind", "say_test");
                o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
                o.put("app_version", MainActivity.APP_VERSION);
                o.put("avatar", avatar != null);
                o.put("synth_ms", new org.json.JSONArray(tts.synthMs));
                o.put("audio_ms", new org.json.JSONArray(tts.audioMs));
                long synth = 0, audio = 0;
                for (long v : tts.synthMs) synth += v;
                for (long v : tts.audioMs) audio += v;
                o.put("rtf", audio == 0 ? JSONObject.NULL : (double) synth / audio);
                o.put("send_to_first_sound_ms", tts.firstFrameNs() == 0 ? JSONObject.NULL : (tts.firstFrameNs() - t0) / 1_000_000);
                o.put("underruns", tts.underruns());
                o.put("dec", dec.isEmpty() ? "cpu" : dec);
                if (tts.error != null) o.put("error", tts.error);
                o.put("thermal", Probe.thermal(this));
                record(o);
                Thread.sleep(1500);
            }
        });
    }

    /** Demo: when nothing is going on, send the next line (null = greeting event), then wait for that turn. */
    private void runScript(List<String> lines, int i) {
        if (isFinishing()) return;
        demoRunning = true;
        boolean busy = prep != PREP_READY || activeRelay != null || activeTts != null || ownMic != null || voice.listening;
        if (busy) { ui.postDelayed(() -> runScript(lines, i), 400); return; }
        if (i >= lines.size()) { demoRunning = false; return; }   // the last turn has finished
        String line = lines.get(i);
        if (line == null) {
            long last = prefs.getLong(k("last_turn_at"), 0);
            sendText(ChatMemory.openEvent(new Date(), last == 0 ? -1 : System.currentTimeMillis() - last, chr.name), -1, "event");
        } else {
            sendText(line, -1, null);
        }
        ui.postDelayed(() -> runScript(lines, i + 1), 2500);   // the turn starts within this, then busy holds it
    }

    /** A demo script is sending lines: hands-free listening stays off in between. */
    private volatile boolean demoRunning;

    /** Keep her voice of each turn for a demo video (set by the adb hook). */
    private volatile boolean recordVoice;

    private void saveRecordedVoice(TtsPipeline tts) {
        try {
            float[] a = tts.keptAudio();
            long ff = tts.firstFrameNs();
            if (a.length == 0 || ff == 0) return;
            long wallStart = System.currentTimeMillis() - (System.nanoTime() - ff) / 1_000_000;
            File dir = new File(getExternalFilesDir(null), "rec");
            if (!dir.isDirectory() && !dir.mkdirs()) return;
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(a.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (float v : a) bb.putFloat(v);
            Files.write(new File(dir, "turn_" + wallStart + ".f32").toPath(), bb.array());
            logEvent("rec", "turn_" + wallStart + " " + a.length / 48 + " ms");
        } catch (Exception e) {
            logEvent("rec", "failed: " + e.getMessage());
        }
    }

    @Override
    protected void onNewIntent(Intent in) {
        super.onNewIntent(in);
        handleAutoIntent(in);
    }
    /** ふぇれ子 in 3D when a ferreko*.vrm is in the model folder (else the 2D picture). */
    private AvatarView avatar;
    /** What the 3D page polls each frame: loading / idle / listen / recognize / think / talk. */
    private volatile String avatarState = "loading";
    private static final int PREP_LOADING = 0, PREP_READY = 1, PREP_FAILED = 2;
    /** Voice (and our own ears, if used) loaded: shown in the state pill instead of a status line. */
    private volatile int prep = PREP_LOADING;
    private volatile String prepError = "";
    private volatile boolean recognizing;
    private TextView hintView, diagView;
    private volatile String lastDiag = "";
    private final Runnable hideHint = () -> hintView.animate().alpha(0).setDuration(400)
            .withEndAction(() -> hintView.setVisibility(View.GONE)).start();

    /** A short message for the user, shown above the input for a few seconds. */
    private void hint(String s) {
        ui.post(() -> {
            hintView.setText(s);
            hintView.animate().cancel();
            hintView.setAlpha(1);
            hintView.setVisibility(View.VISIBLE);
            ui.removeCallbacks(hideHint);
            ui.postDelayed(hideHint, 4000);
        });
    }
    private boolean stickToBottom = true;
    private TextView statePill, bgButton, sendButton;
    private ChatLog chatLog;
    private LinearLayout settingsPanel;

    private String bgLabel() { return "outside".equals(prefs.getString("bg", "room")) ? "外" : "部屋"; }

    private void showBackground() {
        android.graphics.Bitmap b = assetBitmap("outside".equals(prefs.getString("bg", "room")) ? "bg_outside.webp" : "bg_room.webp");
        if (b != null) bgView.setImageBitmap(b);
    }

    private android.graphics.Bitmap assetBitmap(String name) {
        try (java.io.InputStream in = getAssets().open(name)) {
            return android.graphics.BitmapFactory.decodeStream(in);
        } catch (IOException e) {
            return null;
        }
    }

    /** Small glass button for the top bar and the input field. */
    private TextView chip(String text, View.OnClickListener l) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTextColor(0xFFFFFFFF);
        t.setTextSize(14);
        t.setGravity(android.view.Gravity.CENTER);
        t.setPadding(dp(14), dp(7), dp(14), dp(7));
        android.graphics.drawable.GradientDrawable bg = new android.graphics.drawable.GradientDrawable();
        bg.setColor(0x4D000000);
        bg.setStroke(dp(1), 0x40FFFFFF);
        bg.setCornerRadius(dp(18));
        t.setBackground(bg);
        t.setOnClickListener(l);
        return t;
    }

    /** What ふぇれ子 is doing now, in the top bar; also turns 送る into 止める during a turn. */
    private final Runnable stateTick = new Runnable() {
        private String last = "";

        @Override public void run() {
            TtsPipeline t = activeTts;
            boolean listening = ownMic != null || voice != null && voice.listening;
            String s;
            int dot;
            if (avatar != null && t != null && activeCues != null) {
                int idx = t.currentSentence();
                if (idx >= 0 && idx != lastCue) {
                    lastCue = idx;
                    List<JSONObject> cs = activeCues;
                    synchronized (cs) { if (idx < cs.size()) avatar.cue(cs.get(idx).toString()); }
                }
            } else if (t == null) {
                lastCue = -1;
            }
            avatarState = prep == PREP_LOADING ? "loading" : listening ? "listen" : recognizing ? "recognize"
                    : t != null && t.firstFrameNs() != 0 ? "talk" : activeRelay != null || t != null ? "think" : "idle";
            if (prep == PREP_LOADING) {   // the dot blinks while the voice and ears load
                s = "支度中";
                dot = (System.nanoTime() / 600_000_000L) % 2 == 0 ? 0xFFB0B0B8 : 0xFF55555E;
            }
            else if (prep == PREP_FAILED && t == null) { s = "声が出ない"; dot = MicButton.ACCENT; }
            else if (listening) { s = "聞いてる"; dot = MicButton.ACCENT; }
            else if (recognizing) { s = "聞き取り中"; dot = 0xFFF09496; }
            else if (t != null && t.firstFrameNs() != 0) { s = "しゃべり中"; dot = 0xFFFFFFFF; }
            else if (activeRelay != null || t != null) { s = "考え中"; dot = 0xFFB0B0B8; }
            else { s = "待ってる"; dot = 0xFFE8E8EC; }   // bright = ready (支度中 blinks dim)
            String key = s + dot;
            if (!key.equals(last)) {
                last = key;
                android.text.SpannableStringBuilder b = new android.text.SpannableStringBuilder("● " + s);
                b.setSpan(new android.text.style.ForegroundColorSpan(dot), 0, 1, 0);
                statePill.setText(b);
                android.graphics.drawable.GradientDrawable bg = new android.graphics.drawable.GradientDrawable();
                bg.setColor(0x4D000000);
                bg.setCornerRadius(dp(12));
                statePill.setBackground(bg);
                sendButton.setText(activeRelay != null || t != null ? "止める" : "送る");
                micButton.setListening(listening);
            }
            ui.postDelayed(this, 150);
        }
    };

    /** The developer settings (voice, recognizer, model, …) that used to be buttons on the screen. */
    private void buildSettingsPanel() {
        settingsPanel = new LinearLayout(this);
        settingsPanel.setOrientation(LinearLayout.VERTICAL);
        settingsPanel.setPadding(dp(20), dp(8), dp(20), dp(8));

        LinearLayout r1 = row(), r2 = row(), r3 = row();
        r1.addView(button("合言葉", v -> startActivityForResult(
                new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), REQ_TOKEN)));
        modelButton = button(modelLabel(), v -> {
            String[] ms = {"opus", "sonnet", "haiku"};
            int i = java.util.Arrays.asList(ms).indexOf(model());
            prefs.edit().putString("model", ms[(i + 1) % ms.length]).apply();
            modelButton.setText(modelLabel());
            if (tokenFile().isFile()) warmCache();
        });
        r1.addView(modelButton);
        asrButton = button(asrLabel(), v -> {
            prefs.edit().putString("asr", "android".equals(prefs.getString("asr", "")) ? "own" : "android").apply();
            asrButton.setText(asrLabel());
            if (ownAsr() && !LocalAsr.isLoaded()) new Thread(() -> {
                try { LocalAsr.get(packDir()); setStatus("耳の準備OK"); } catch (Exception e) { hint("耳の準備に失敗: " + e.getMessage()); }
            }, "asr-preload").start();
        });
        r1.addView(asrButton);
        ditButton = button(ditLabel(), v -> {
            String[] vs = Engines.ditVariants(packDir());
            int i = java.util.Arrays.asList(vs).indexOf(dit());
            prefs.edit().putString("dit", vs[(i + 1) % vs.length]).apply();
            ditButton.setText(ditLabel());
        });
        r2.addView(ditButton);
        decButton = button(decLabel(), v -> {
            String[] vs = Engines.decVariants(packDir());
            int i = java.util.Arrays.asList(vs).indexOf(dec());
            prefs.edit().putString("decoder", vs[(i + 1) % vs.length]).apply();
            decButton.setText(decLabel());
        });
        r2.addView(decButton);
        refButton = button(refLabel(), v -> {
            prefs.edit().putBoolean(k("use_ref"), !prefs.getBoolean(k("use_ref"), true)).apply();
            refButton.setText(refLabel());
        });
        r2.addView(refButton);
        speedButton = button(speedLabel(), v -> {
            float[] ss = {1.0f, 1.1f, 1.15f, 1.2f, 1.3f};
            float cur = speed();
            int i = 0;
            while (i < ss.length && ss[i] <= cur + 0.001f) i++;
            prefs.edit().putFloat("speed", ss[i % ss.length]).apply();
            speedButton.setText(speedLabel());
        });
        r3.addView(speedButton);
        r3.addView(button("使用額", v -> job(this::showUsage)));
        r3.addView(button("記憶", v -> showMemo()));
        LinearLayout r0 = row();
        r0.addView(button("キャラ:" + chr.name, v -> chooseCharacter()));
        settingsPanel.addView(r0, 0);
        r3.addView(button("好感度", v -> showAffection()));
        Button[] greet = {null};
        greet[0] = button(greetingLabel(), v -> {
            prefs.edit().putBoolean("greeting", !prefs.getBoolean("greeting", true)).apply();
            greet[0].setText(greetingLabel());
        });
        r3.addView(greet[0]);
        if (chr.vrm(packDir()) != null) {
            r3.addView(button(prefs.getBoolean("avatar3d", true) ? "姿:3D" : "姿:2D", v -> {
                prefs.edit().putBoolean("avatar3d", !prefs.getBoolean("avatar3d", true)).apply();
                recreate();
            }));
        }
        r3.addView(button("試験画面", v -> startActivity(new Intent(this, MainActivity.class))));
        r3.addView(button("最初から", v -> confirmReset()));
        settingsPanel.addView(r1);
        settingsPanel.addView(r2);
        settingsPanel.addView(r3);

        LinearLayout capRow = row();
        TextView capLabel = new TextView(this);
        capLabel.setText("声");
        capLabel.setPadding(dp(4), 0, dp(4), 0);
        capRow.addView(capLabel);
        captionInput = new EditText(this);
        captionInput.setSingleLine(true);
        // When a new build ships a different default caption (chat/voice.txt), switch to it once.
        String defCaption = chr.defaultCaption;
        if (!defCaption.equals(prefs.getString(k("caption_default"), ""))) {
            prefs.edit().putString(k("caption"), defCaption).putString(k("caption_default"), defCaption).apply();
        }
        captionInput.setText(prefs.getString(k("caption"), defCaption));
        capRow.addView(captionInput, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        settingsPanel.addView(capRow);

        handsFree = new android.widget.CheckBox(this);
        handsFree.setText("会話モード（ふぇれ子が話し終わったら自動で聞き取る）");
        handsFree.setChecked(prefs.getBoolean("hands_free", false));
        handsFree.setOnCheckedChangeListener((b2, on) -> prefs.edit().putBoolean("hands_free", on).apply());
        settingsPanel.addView(handsFree);
        bargeInBox = new android.widget.CheckBox(this);
        bargeInBox.setText("割り込み（話している途中でも声をかけたら止まる）");
        bargeInBox.setChecked(prefs.getBoolean("barge_in", true));
        bargeInBox.setOnCheckedChangeListener((b2, on) -> prefs.edit().putBoolean("barge_in", on).apply());
        settingsPanel.addView(bargeInBox);

        diagView = new TextView(this);
        diagView.setTypeface(Typeface.MONOSPACE);
        diagView.setTextSize(10);
        diagView.setTextIsSelectable(true);
        diagView.setPadding(0, dp(12), 0, 0);
        diagView.setText(lastDiag);
        settingsPanel.addView(diagView);
    }

    private void openSettings() {
        if (settingsPanel.getParent() != null) ((android.view.ViewGroup) settingsPanel.getParent()).removeView(settingsPanel);
        ScrollView sv = new ScrollView(this);
        sv.addView(settingsPanel);
        new AlertDialog.Builder(this)
                .setTitle("設定")
                .setView(sv)
                .setPositiveButton("閉じる", (d, w) -> prefs.edit().putString(k("caption"), captionInput.getText().toString().trim()).apply())
                .show();
    }

    private LinearLayout row() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.HORIZONTAL);
        return l;
    }

    private Button button(String text, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setOnClickListener(l);
        b.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        buttons.add(b);
        return b;
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    private void append(String s) {
        ui.post(() -> {
            chatLog.add(s);
        });
    }

    private void appendRaw(String s) {
        ui.post(() -> {
            chatLog.addRaw(s);
        });
    }

    /** Technical record (load times, the last turn's timings): shown at the bottom of the settings. */
    private void setStatus(String s) {
        lastDiag = s;
        ui.post(() -> {
            if (diagView != null) diagView.setText(s);
        });
    }

    private void show(Map<String, Object> m) {
        Object c = m.get("content");
        if (c instanceof String && ((String) c).startsWith(ChatMemory.EVENT_OPEN)) return;   // not something the user said
        String text;
        if (c instanceof String) {
            text = (String) c;
        } else {
            StringBuilder b = new StringBuilder();
            for (Object o : MiniJson.arr(c)) {
                Map<String, Object> blk = MiniJson.obj(o);
                if ("text".equals(blk.get("type"))) b.append((String) blk.get("text"));
            }
            text = b.toString();
        }
        append(("user".equals(m.get("role")) ? "あなた：" : "ふぇれ子：") + Affection.clean(text) + "\n");
    }

    private interface Job { void run() throws Exception; }

    private void job(Job j) {
        cancel = false;
        for (Button b : buttons) b.setEnabled(false);
        worker.execute(() -> {
            try {
                j.run();
            } catch (Throwable t) {
                append("（エラー: " + t.getMessage() + "）");
            } finally {
                ui.post(() -> { for (Button b : buttons) b.setEnabled(true); });
            }
        });
    }

    // ---------- settings / files ----------

    private File packDir() { return new File(getFilesDir(), "pack"); }

    private File tokenFile() { return new File(getFilesDir(), "relay-token.txt"); }

    private File historyFile() { return chr.historyFile(); }

    private File memoFile() { return chr.memoFile(); }

    private String readMemo() {
        try {
            File f = memoFile();
            return f.isFile() ? new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8) : "";
        } catch (IOException e) {
            return "";
        }
    }

    private void writeMemo(String m) throws IOException {
        File f = memoFile();
        File dir = f.getParentFile();
        if (dir != null && !dir.isDirectory() && !dir.mkdirs()) throw new IOException("mkdir " + dir);
        Files.write(f.toPath(), m.getBytes(StandardCharsets.UTF_8));
    }

    /** ふぇれ子's system prompt: persona + what she remembers + today's date (changes once a day). */
    private String systemPrompt() {
        String today = new SimpleDateFormat("yyyy年M月d日（E）", Locale.JAPAN).format(new Date());
        return ChatMemory.systemPrompt(persona, memo, today);
    }

    /**
     * When the history has grown past ChatMemory.COMPACT_AT messages, fold the older part into the
     * memo (Sonnet, ~7 s, ~$0.02, in the background) and keep only the recent turns. The folded
     * messages go to history-archive.jsonl, so nothing is lost.
     */
    private void maybeCompact() {
        List<Object> snap = new ArrayList<>(history);
        int cut = ChatMemory.cutIndex(snap);
        if (cut < 0 || compacting) return;
        compacting = true;
        String oldMemo = memo;
        new Thread(() -> {
            try {
                long t = System.nanoTime();
                RelayClient.Reply r = relay().chat(ChatMemory.summarizer(chr.name),
                        ChatMemory.requestMessages(oldMemo, ChatMemory.transcript(snap.subList(0, cut), chr.name)), "sonnet", null);
                String next = ChatMemory.cleanMemo(r.text());
                if (next.isEmpty()) throw new IOException("empty memo");
                long ms = (System.nanoTime() - t) / 1_000_000;
                worker.execute(() -> {   // after any turn in progress, so the history isn't changed under it
                    try {
                        List<Object> h = history;
                        if (h.size() < cut || h.get(0) != snap.get(0)) return;   // reset meanwhile
                        StringBuilder arch = new StringBuilder();
                        for (Object o : h.subList(0, cut)) arch.append(MiniJson.write(o)).append('\n');
                        Files.write(chr.archiveFile().toPath(),
                                arch.toString().getBytes(StandardCharsets.UTF_8),
                                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
                        writeMemo(next);
                        memo = next;
                        history = new ArrayList<>(h.subList(cut, h.size()));
                        saveHistory();
                        logEvent("memory", "folded " + cut + " messages into " + next.length() + " chars in " + ms + " ms");
                        warmCache();   // the system prompt changed
                    } catch (Exception e) {
                        logEvent("memory", "apply failed: " + e.getMessage());
                    } finally {
                        compacting = false;
                    }
                });
            } catch (Exception e) {
                logEvent("memory", "summary failed: " + e.getMessage());
                compacting = false;
            }
        }, "memory").start();
    }

    /** ふぇれ子's affection for キミ (0..100, starts at 0: the user wanted it earned, 2026-10-10). */
    private int affection() { return prefs.getInt(k("aff_value"), 0); }

    /**
     * Applies the change Claude asked for at the end of its reply (［好感度:+N］), clamped per reply
     * and per day; shows a small ♡ hint when it moves.
     * @return the change applied
     */
    private int applyAffection(String replyText) {
        Integer req = Affection.requested(replyText);
        if (req == null) return 0;
        String today = new SimpleDateFormat("yyyy-MM-dd", Locale.US).format(new Date());
        int gained = today.equals(prefs.getString(k("aff_day"), "")) ? prefs.getInt(k("aff_gained"), 0) : 0;
        int before = affection();
        int d = Affection.allowed(req, gained);
        int after = Math.max(0, Math.min(100, before + d));
        d = after - before;
        prefs.edit().putInt(k("aff_value"), after).putString(k("aff_day"), today)
                .putInt(k("aff_gained"), gained + Math.max(0, d)).apply();
        logEvent("affection", "requested " + req + ", applied " + d + ", now " + after);
        if (d != 0) {
            String stage = Affection.stage(after);
            hint((d > 0 ? "♡ +" + d : "♡ " + d) + (stage.equals(Affection.stage(before)) ? "" : "（" + stage + "）"));
        }
        return d;
    }

    private void showAffection() {
        int v = affection();
        new AlertDialog.Builder(this)
                .setTitle("好感度")
                .setMessage(v + " / 100（" + Affection.stage(v) + "）\n\n1回で上がるのは最大 +" + Affection.MAX_UP
                        + "、1日に上がるのは最大 +" + Affection.DAILY_UP + "。")
                .setPositiveButton("閉じる", null)
                .setNegativeButton("0 に戻す", (d, w) -> new AlertDialog.Builder(this)
                        .setMessage("好感度を 0 に戻す？")
                        .setPositiveButton("戻す", (d2, w2) -> prefs.edit().putInt(k("aff_value"), 0).putInt(k("aff_gained"), 0).apply())
                        .setNegativeButton("やめる", null)
                        .show())
                .show();
    }

    /** No greeting if they talked less than this long ago (reopening the app right away). */
    private static final long GREETING_GAP_MS = 30 * 60 * 1000L;

    /**
     * ふぇれ子 speaks first when the app is opened (after a while): one turn with an event message
     * (time of day, how long since they last talked) that is not shown as the user's words.
     * Waits until the voice is loaded so she starts speaking right away. ~$0.003 with a warm cache.
     * @return true if a greeting was scheduled
     */
    private boolean scheduleGreeting() {
        if (!prefs.getBoolean("greeting", true) || !new File(packDir(), "pack.json").isFile()) return false;
        long last = prefs.getLong(k("last_turn_at"), 0);
        // builds before 0.15 didn't record it: the history file was last written after the last turn
        if (last == 0 && !history.isEmpty() && historyFile().isFile()) last = historyFile().lastModified();
        long now = System.currentTimeMillis();
        if (last != 0 && now - last < GREETING_GAP_MS) return false;
        String text = ChatMemory.openEvent(new Date(now), last == 0 ? -1 : now - last, chr.name);
        long until = now + 90_000;
        Runnable[] wait = {null};
        wait[0] = () -> {
            if (isFinishing() || isDestroyed()) return;
            if (prep == PREP_READY && activeRelay == null && activeTts == null && ownMic == null && !voice.listening) {
                logEvent("greeting", text);
                sendText(text, -1, "event");
            } else if (prep != PREP_FAILED && System.currentTimeMillis() < until) {
                ui.postDelayed(wait[0], 300);
            }
        };
        ui.postDelayed(wait[0], 300);
        return true;
    }

    private String greetingLabel() { return prefs.getBoolean("greeting", true) ? "あいさつ:する" : "あいさつ:しない"; }

    /**
     * Switch who to talk to: ふぇれ子 or a character folder in files/chars/ (see CharacterProfile).
     * Each keeps its own conversation, memo and affection.
     */
    private void chooseCharacter() {
        List<CharacterProfile> all = CharacterProfile.all(this);
        String[] names = new String[all.size()];
        int cur = 0;
        for (int i = 0; i < names.length; i++) {
            names[i] = all.get(i).name;
            if (all.get(i).id.equals(chr.id)) cur = i;
        }
        new AlertDialog.Builder(this)
                .setTitle("話す相手")
                .setSingleChoiceItems(names, cur, (d, which) -> {
                    d.dismiss();
                    CharacterProfile next = all.get(which);
                    if (next.id.equals(chr.id)) return;
                    prefs.edit().putString("character", next.id).apply();
                    logEvent("character", next.id);
                    recreate();
                })
                .setNegativeButton("やめる", null)
                .show();
    }

    /** Shows what ふぇれ子 remembers, with a way to forget it. */
    private void showMemo() {
        TextView tv = new TextView(this);
        tv.setText(memo.trim().isEmpty() ? "まだ何も覚えてない。会話が長くなると、古いほうからここにまとめるよ。" : memo.trim());
        tv.setTextIsSelectable(true);
        tv.setPadding(dp(20), dp(8), dp(20), dp(8));
        ScrollView sv = new ScrollView(this);
        sv.addView(tv);
        new AlertDialog.Builder(this)
                .setTitle(chr.name + "が覚えていること")
                .setView(sv)
                .setPositiveButton("閉じる", null)
                .setNegativeButton("忘れさせる", (d, w) -> new AlertDialog.Builder(this)
                        .setMessage("覚えていることを全部消す？（会話の記録は残るよ）")
                        .setPositiveButton("消す", (d2, w2) -> worker.execute(() -> {
                            memo = "";
                            File f = memoFile();
                            if (f.isFile() && !f.delete()) logEvent("memory", "delete failed");
                            warmCache();
                            append("（覚えていたことを消したよ）");
                        }))
                        .setNegativeButton("やめる", null)
                        .show())
                .show();
    }

    /** int8dyn by default when present: ~15% faster on the phone than FP32, same voice by ear. */
    private String dit() {
        String def = new File(packDir(), "dit_step.int8dyn.onnx").isFile() ? "int8dyn" : "";
        return prefs.getString("dit", def);
    }

    /**
     * GPU 16bit by default when the GPU model is installed: RTF 0.39 vs 1.24 on the CPU, no audible
     * difference (user, 2026-10-09). With the 36-frame windows (dec36_f32.tflite) each GPU job is
     * half as long, so the 3D view stalls 0.35 s instead of 0.7 s while she speaks (RTF ~0.6); the
     * user chose that as the default (2026-10-10). A new default replaces the stored one once.
     */
    private String dec() {
        String def = new File(packDir(), "dec36_f32.tflite").isFile() ? "gpu16@36"
                : new File(packDir(), "dec76_f32.tflite").isFile() ? "gpu16" : "";
        if (!def.equals(prefs.getString("decoder_default", ""))) {
            prefs.edit().putString("decoder", def).putString("decoder_default", def).apply();
        }
        return prefs.getString("decoder", def);   // new key: older builds stored CPU/fp16 under "dec"
    }

    private String refLabel() {
        if (!new File(chr.voiceDir(packDir()), "speaker.f32").isFile()) return "声:未登録";
        return prefs.getBoolean(k("use_ref"), true) ? "声:登録" : "声:説明だけ";
    }

    /** Claude model for replies: opus (best persona), sonnet / haiku (they can skip thinking, so they start sooner). */
    private String model() { return prefs.getString("model", "opus"); }

    private String modelLabel() {
        String m = model();
        return "頭:" + Character.toUpperCase(m.charAt(0)) + m.substring(1);
    }

    /** ふぇれ子's speaking rate: 1.2 chosen by the user after listening to 1.0-1.3 (2026-10-09). */
    private static final float DEFAULT_SPEED = 1.2f;

    private float speed() {
        // a new default replaces the old one once, as with the voice caption
        if (prefs.getFloat("speed_default", 0f) != DEFAULT_SPEED) {
            prefs.edit().putFloat("speed", DEFAULT_SPEED).putFloat("speed_default", DEFAULT_SPEED).apply();
        }
        return prefs.getFloat("speed", DEFAULT_SPEED);
    }

    private String speedLabel() { return String.format(Locale.US, "テンポ:%.2f", speed()); }

    private String asrLabel() {
        if (!LocalAsr.available(packDir())) return "耳:Android";
        return ownAsr() ? "耳:自前" : "耳:Android";
    }

    private String decLabel() {
        String d = dec();
        return "Dec:" + (d.isEmpty() ? "CPU" : d);
    }

    private String ditLabel() {
        String d = dit();
        return "DiT:" + (d.isEmpty() ? "FP32" : d);
    }

    private String asset(String name) {
        try (InputStream in = getAssets().open(name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            return "";
        }
    }

    private List<Object> loadHistory() {
        try {
            File f = historyFile();
            if (f.isFile()) return MiniJson.arr(MiniJson.parse(new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8)));
        } catch (Exception e) {
            append("（会話の記録を読めなかった: " + e.getMessage() + "）");
        }
        return new ArrayList<>();
    }

    private void saveHistory() throws IOException {
        File f = historyFile();
        File dir = f.getParentFile();
        if (dir != null && !dir.isDirectory() && !dir.mkdirs()) throw new IOException("mkdir " + dir);
        File tmp = new File(f.getPath() + ".tmp");
        Files.write(tmp.toPath(), MiniJson.write(history).getBytes(StandardCharsets.UTF_8));
        if (!tmp.renameTo(f)) throw new IOException("rename history");
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req != REQ_TOKEN || res != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        job(() -> {
            String all;
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                if (in == null) throw new IOException("開けなかった");
                byte[] buf = new byte[2048];
                int n = in.read(buf);
                all = new String(buf, 0, Math.max(n, 0), StandardCharsets.UTF_8);
            }
            // one value per line: the relay's address (https://…) and/or the passphrase
            String url = null, token = null;
            for (String line : all.split("\\r?\\n")) {
                String t = line.trim();
                if (t.isEmpty()) continue;
                if (t.startsWith("https://")) {
                    if (!RelayClient.validUrl(t)) throw new IOException("中継の住所の形じゃないみたい（https://… の行）");
                    url = t;
                } else if (t.matches("[A-Za-z0-9_\\-]{20,200}")) {
                    token = t;
                } else {
                    throw new IOException("合言葉か中継の住所の形じゃない行がある（relay-token.txt を選んでね）");
                }
            }
            if (url == null && token == null) throw new IOException("中身が空みたい");
            if (url != null) Files.write(relayUrlFile().toPath(), url.getBytes(StandardCharsets.UTF_8));
            if (token != null) Files.write(tokenFile().toPath(), token.getBytes(StandardCharsets.UTF_8));
            append("（" + (url != null && token != null ? "中継の住所と合言葉" : url != null ? "中継の住所" : "合言葉")
                    + "を取り込んだ。ダウンロードフォルダのファイルは消しておいてね）");
        });
    }

    private RelayClient relay() throws IOException {
        if (!tokenFile().isFile()) throw new IOException("合言葉がまだ取り込まれてないよ");
        if (!relayUrlFile().isFile()) throw new IOException("中継の住所がまだ入ってないよ（設定の「合言葉」で取り込んでね）");
        String token = new String(Files.readAllBytes(tokenFile().toPath()), StandardCharsets.UTF_8).trim();
        String url = new String(Files.readAllBytes(relayUrlFile().toPath()), StandardCharsets.UTF_8).trim();
        return new RelayClient(url, token);
    }

    /** The relay's address (the user's own Cloudflare Worker), kept next to the passphrase; not in the code. */
    private File relayUrlFile() { return new File(getFilesDir(), "relay-url.txt"); }

    private void showUsage() throws Exception {
        Map<String, Object> u = relay().usage();
        append(String.format(Locale.US, "（今日の使用額 $%.4f / 上限 $%.2f、%d 回）",
                ((Number) u.get("usd")).doubleValue(), ((Number) u.get("limit_usd")).doubleValue(), ((Number) u.get("requests")).intValue()));
    }

    /** The conversation starts over; the old one is kept as history-<time>.json (memo and affection stay). */
    private void resetHistory() throws IOException {
        File f = historyFile();
        if (f.isFile() && !f.renameTo(new File(f.getParentFile(),
                "history-" + new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date()) + ".json"))) {
            throw new IOException("rename");
        }
        history = new ArrayList<>();
        ui.post(() -> chatLog.clear());
    }

    private void confirmReset() {
        new AlertDialog.Builder(this)
                .setMessage("会話を最初からにする？（今までの会話は別名で残すよ。覚えていることのメモはそのまま）")
                .setPositiveButton("最初から", (d, w) -> job(() -> {
                    resetHistory();
                    append("（最初からにしたよ）");
                }))
                .setNegativeButton("やめる", null)
                .show();
    }

    // ---------- one turn ----------

    private void talk(String userText, long sttMs, String sttMode) throws Exception {
        long t0 = System.nanoTime();
        RelayClient relay = relay();
        String caption = prefs.getString(k("caption"), "");   // saved from the field when 送信 was pressed
        String dit = dit();
        String dec = dec();
        String model = model();
        String system = systemPrompt();
        float speed = speed();
        recognizing = false;
        boolean event = "event".equals(sttMode);   // e.g. the app was opened: not shown as the user's words
        if (!event) append("あなた：" + userText + "\n");

        // Load the voice (first turn only, ~5 s) while the relay request is in flight.
        long[] engineReady = {0};
        java.util.concurrent.FutureTask<IrodoriEngine> engineTask = new java.util.concurrent.FutureTask<>(() -> {
            IrodoriEngine e = Engines.get(packDir(), THREADS, IrodoriEngine.Accel.NONE, dit, dec);
            e.speed = speed;
            e.setReference(chr.voiceDir(packDir()));
            engineReady[0] = System.nanoTime();
            return e;
        });
        new Thread(engineTask, "engine-load").start();

        Map<String, Object> userMsg = new java.util.LinkedHashMap<>();
        userMsg.put("role", "user");
        int affBefore = affection();
        userMsg.put("content", (interruptedLast ? "（" + chr.name + "が話している途中で割り込んだ）" + userText : userText)
                + "\n" + Affection.note(affBefore));
        interruptedLast = false;
        List<Object> request = new ArrayList<>(history);
        request.add(userMsg);

        boolean useRef = prefs.getBoolean(k("use_ref"), true);
        TtsPipeline tts = new TtsPipeline(engineTask, caption, 0, t0, useRef);
        tts.keepAudio = recordVoice;
        boolean[] bargedIn = {false};
        BargeInDetector barge = null;
        if (bargeInBox.isChecked()
                && checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            BargeInDetector[] self = {null};
            boolean own = ownAsr();
            barge = new BargeInDetector(BargeInDetector.Mode.BARGE_IN, () -> tts.firstFrameNs() != 0, new BargeInDetector.Listener() {
                @Override public void bargeIn() {
                    bargedIn[0] = true;
                    AvatarView av = avatar;
                    if (av != null) ui.post(av::interrupt);
                    tts.cancel();
                    RelayClient r = activeRelay;
                    if (r != null) r.cancel();
                    if (own) {
                        logEvent("barge_in", "own recognizer");
                        setStatus("聞いてるよ…");
                    } else {
                        ui.post(() -> handOff(self[0]));
                    }
                }
                @Override public void utterance(float[] pcm, int n) {
                    if (own) recognizeAndSend(pcm, n, true);
                }
            });
            self[0] = barge;
        }
        SentenceSplitter splitter = new SentenceSplitter();
        Affection.TagFilter tags = new Affection.TagFilter();
        List<JSONObject> cues = new ArrayList<>();          // per said sentence, for the 3D face and motion
        List<String> pendingActions = new ArrayList<>();
        int[] tagsSeen = {0};
        // a sentence goes to the voice together with its cue (emotion from the emoji, actions from ［…］ tags)
        java.util.function.Consumer<String> say = s -> {
            synchronized (cues) {
                for (; tagsSeen[0] < tags.tags.size(); tagsSeen[0]++) {
                    String tg = tags.tags.get(tagsSeen[0]);
                    String inner = tg.substring(1, tg.length() - 1).trim();
                    if (!inner.startsWith("好感度")) pendingActions.add(inner);
                }
                JSONObject c = new JSONObject();
                try {
                    c.put("text", s);
                    c.put("emotion", emotionOf(s));
                    org.json.JSONArray acts = new org.json.JSONArray(pendingActions);
                    if (cues.isEmpty() && event && acts.length() == 0) acts.put("手を振る");   // greeting
                    c.put("actions", acts);
                    c.put("first", cues.isEmpty());
                } catch (Exception ignored) { }
                pendingActions.clear();
                cues.add(c);
                tts.say(s);
            }
        };
        activeCues = cues;
        long[] lastDelta = {0};
        // a sentence end followed by a pause in the stream: speak it without waiting for more text
        java.util.concurrent.ScheduledExecutorService idle = java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
        idle.scheduleWithFixedDelay(() -> {
            long t = lastDelta[0];
            if (t != 0 && System.nanoTime() - t > IDLE_RELEASE_NS) for (String s : splitter.releaseEnded()) say.accept(s);
        }, 50, 50, java.util.concurrent.TimeUnit.MILLISECONDS);
        long[] firstText = {0};
        activeRelay = relay;
        activeTts = tts;
        appendRaw("ふぇれ子：");
        setStatus("考え中…");
        RelayClient.Reply reply;
        try {
            reply = relay.chat(system, request, model, delta -> {
                if (firstText[0] == 0) firstText[0] = System.nanoTime();
                String visible = tags.feed(delta);   // ［好感度:+1］ etc. are for the app only
                if (visible.isEmpty()) return;
                appendRaw(visible);
                lastDelta[0] = System.nanoTime();
                for (String s : splitter.feed(visible)) say.accept(s);
            });
            idle.shutdownNow();
            String rest = tags.flush();
            if (!rest.isEmpty()) {
                appendRaw(rest);
                splitter.feed(rest);
            }
            for (String s : splitter.flush()) say.accept(s);
        } catch (IOException e) {
            tts.cancel();
            activeRelay = null;
            activeTts = null;
            if (barge != null) barge.stop();
            if (bargedIn[0]) {
                // cut off while the reply was still streaming: keep the user's message, drop the partial reply
                append("\n（割り込み）\n");
                history = request;
                saveHistory();
                interruptedLast = true;
                return;
            }
            append(cancel ? "\n（止めた）\n" : "\n（" + relayHint(e) + "）\n");
            return;
        } finally {
            idle.shutdownNow();
            tts.end();
        }
        append("\n");
        setStatus("しゃべり中…");
        tts.await();
        if (recordVoice) saveRecordedVoice(tts);
        if (barge != null) barge.stop();
        activeRelay = null;
        activeTts = null;
        if (bargedIn[0]) {
            append("（割り込み）\n");
            interruptedLast = true;
        }

        if ("refusal".equals(reply.stopReason)) {
            append("（この話題は答えられないって返ってきた。記録には入れないね）\n");
            return;
        }
        Map<String, Object> asst = new java.util.LinkedHashMap<>();
        asst.put("role", "assistant");
        asst.put("content", reply.content);   // exactly as streamed: thinking signatures included
        request.add(asst);
        history = request;
        saveHistory();
        prefs.edit().putLong(k("last_turn_at"), System.currentTimeMillis()).apply();
        int affDelta = applyAffection(reply.text());
        maybeCompact();

        JSONObject o = new JSONObject();
        o.put("kind", "chat");
        o.put("time", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
        o.put("app_version", MainActivity.APP_VERSION);
        o.put("dit", dit.isEmpty() ? "fp32" : dit);
        o.put("dec", dec.isEmpty() ? "fp32" : dec);
        o.put("model_choice", model);
        o.put("affection", affection());
        o.put("affection_delta", affDelta);
        if (event) o.put("event", userText.substring(0, Math.min(40, userText.length())));
        o.put("speed", speed);
        o.put("caption", !caption.isEmpty());
        o.put("reference", useRef && new File(chr.voiceDir(packDir()), "speaker.f32").isFile());
        o.put("character", chr.id);
        if (sttMode != null) {
            o.put("stt_mode", sttMode);
            o.put("stt_ms", sttMs);
        }
        if (barge != null) {
            o.put("barge_in", bargedIn[0]);
            o.put("barge_aec", barge.aec);
            o.put("barge_echo_db", Double.isNaN(barge.echoDb) ? JSONObject.NULL : barge.echoDb);
            o.put("barge_threshold_db", Double.isNaN(barge.thresholdDb) ? JSONObject.NULL : barge.thresholdDb);
            o.put("barge_max_db", barge.maxDb);
            if (barge.error != null) o.put("barge_error", barge.error);
        }
        o.put("engine_load_ms", engineReady[0] == 0 ? JSONObject.NULL : (engineReady[0] - t0) / 1_000_000);
        o.put("send_to_first_text_ms", firstText[0] == 0 ? JSONObject.NULL : (firstText[0] - t0) / 1_000_000);
        o.put("send_to_first_sentence_ms", tts.firstSentenceQueuedNs == 0 ? JSONObject.NULL : (tts.firstSentenceQueuedNs - t0) / 1_000_000);
        o.put("send_to_first_sound_ms", tts.firstFrameNs() == 0 ? JSONObject.NULL : (tts.firstFrameNs() - t0) / 1_000_000);
        o.put("total_ms", (System.nanoTime() - t0) / 1_000_000);
        o.put("sentences", tts.synthMs.size());
        o.put("synth_ms", new org.json.JSONArray(tts.synthMs));
        o.put("audio_ms", new org.json.JSONArray(tts.audioMs));
        o.put("underruns", tts.underruns());
        o.put("reply_chars", reply.text().length());
        o.put("model", reply.model);
        o.put("usage", new JSONObject(MiniJson.write(reply.usage)));
        o.put("thermal", Probe.thermal(this));
        record(o);
        setStatus(MainActivity.summaryLine(o));
        if (tts.error != null) append("（声でエラー: " + tts.error + "）");
        // hands-free: listen again once ふぇれ子 has finished speaking; after a barge-in, always listen
        if (!bargedIn[0] && handsFree.isChecked() && !cancel && !demoRunning) ui.post(this::startListening);
    }

    private static String relayHint(IOException e) {
        if (e instanceof RelayClient.RelayException) {
            int s = ((RelayClient.RelayException) e).status;
            if (s == 401) return "合言葉が合わないみたい";
            if (s == 429) return "今日の上限に達したみたい";
            if (s == 503) return "中継に API キーが入ってないみたい";
        }
        return "中継につながらないみたい: " + e.getMessage();
    }

    private void record(JSONObject o) throws IOException {
        try (FileWriter w = new FileWriter(new File(getExternalFilesDir(null), "runs.jsonl"), StandardCharsets.UTF_8, true)) {
            w.write(o.toString());
            w.write("\n");
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (avatar != null) avatar.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (avatar != null) avatar.onResume();
    }

    @Override
    protected void onDestroy() {
        if (avatar != null) avatar.destroy();
        ui.removeCallbacks(stateTick);
        if (voice != null) voice.destroy();
        BargeInDetector m = ownMic;
        if (m != null) m.abort();
        cancel = true;
        RelayClient r = activeRelay;
        if (r != null) r.cancel();
        TtsPipeline t = activeTts;
        if (t != null) t.cancel();
        worker.shutdownNow();
        super.onDestroy();
    }
}
