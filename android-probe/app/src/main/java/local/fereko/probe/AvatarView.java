package local.fereko.probe;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Color;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.function.Consumer;
import java.util.function.DoubleSupplier;
import java.util.function.Supplier;

/**
 * ふぇれ子 in 3D (web/avatar.html: three.js + three-vrm) in a transparent WebView. Every request is
 * answered by the app: https://fereko.local/web/... from the APK's assets, /pack/... (the VRM, which
 * is the author's own and is not shipped in the APK) from the model folder; anything else is refused,
 * so the page never touches the network.
 */
@SuppressLint("ViewConstructor")
final class AvatarView extends WebView {
    static final String HOST = "fereko.local";

    /** The model to show: the newest ferreko*.vrm in the pack, or null. */
    static File findVrm(File pack) {
        File[] fs = pack.listFiles((d, n) -> n.startsWith("ferreko") && n.endsWith(".vrm"));
        if (fs == null || fs.length == 0) return null;
        File best = fs[0];
        for (File f : fs) if (f.lastModified() > best.lastModified()) best = f;
        return best;
    }

    @SuppressLint("SetJavaScriptEnabled")
    /** Called (on a WebView thread) when a VRMA reaction starts: its name and length in seconds. */
    interface MotionListener { void started(String name, double seconds); }

    AvatarView(Context c, File pack, File vrm, Supplier<String> state, DoubleSupplier level, Consumer<String> log,
               MotionListener motion) {
        super(c);
        setBackgroundColor(Color.TRANSPARENT);
        setVerticalScrollBarEnabled(false);
        setHorizontalScrollBarEnabled(false);
        WebSettings s = getSettings();
        s.setJavaScriptEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setCacheMode(WebSettings.LOAD_NO_CACHE);
        addJavascriptInterface(new Object() {
            @JavascriptInterface public String state() { return state.get(); }
            @JavascriptInterface public double level() { return level.getAsDouble(); }
            @JavascriptInterface public void log(String json) { log.accept(json); }
            @JavascriptInterface public void motionStarted(String name, double seconds) { motion.started(name, seconds); }
            /** the motions (.vrma) in the model folder, as a JSON array of file names */
            @JavascriptInterface public String anims() {
                String[] fs = pack.list((d, n) -> n.endsWith(".vrma"));
                StringBuilder b = new StringBuilder("[");
                if (fs != null) for (String f : fs) b.append(b.length() > 1 ? "," : "").append('"').append(f).append('"');
                return b.append(']').toString();
            }
        }, "Fereko");
        setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView v, WebResourceRequest r) {
                if (!HOST.equals(r.getUrl().getHost())) return refuse();
                String path = r.getUrl().getPath();
                if (path == null || path.contains("..")) return refuse();
                try {
                    InputStream in;
                    if (path.startsWith("/web/")) {
                        in = getContext().getAssets().open(path.substring(1));
                    } else if (path.startsWith("/pack/")) {
                        File f = new File(pack, path.substring(6));
                        if (!f.getParentFile().equals(pack) || !(f.getName().endsWith(".vrm") || f.getName().endsWith(".vrma")
                                || f.getName().endsWith(".json"))) return refuse();
                        in = new FileInputStream(f);
                    } else {
                        return refuse();
                    }
                    return new WebResourceResponse(mime(path), mime(path).startsWith("text") ? "utf-8" : null, in);
                } catch (IOException e) {
                    return new WebResourceResponse("text/plain", "utf-8", 404, "Not Found", null, null);
                }
            }
        });
        loadUrl("https://" + HOST + "/web/avatar.html?vrm=" + android.net.Uri.encode("/pack/" + vrm.getName()));
    }

    /** A sentence starts playing: {"emotion": "😏", "actions": ["うなずく"], "text": "…", "first": true}. */
    void cue(String json) { evaluateJavascript("window.avatarCue && window.avatarCue(" + json + ")", null); }

    /** The user talked over her. */
    void interrupt() { evaluateJavascript("window.avatarInterrupt && window.avatarInterrupt()", null); }

    /** Plays a VRMA reaction by name (e.g. "turn_around"); ignored while one is playing. */
    void playReaction(String name) {
        if (name.matches("[A-Za-z0-9_]+")) evaluateJavascript("window.playReaction && window.playReaction('" + name + "')", null);
    }

    private static WebResourceResponse refuse() {
        return new WebResourceResponse("text/plain", "utf-8", 403, "Forbidden", null, null);
    }

    private static String mime(String path) {
        if (path.endsWith(".html")) return "text/html";
        if (path.endsWith(".js") || path.endsWith(".mjs")) return "text/javascript";
        if (path.endsWith(".json")) return "application/json";
        if (path.endsWith(".txt")) return "text/plain";
        return "application/octet-stream";   // .vrm / .vrma (glTF binary)
    }
}
