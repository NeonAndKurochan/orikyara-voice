package local.fereko.probe.core;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Client for the ふぇれ子 relay (Cloudflare Worker). Same logic as tools/relay_client.py:
 * the relay passes Claude's SSE stream through, and the assistant content blocks are rebuilt
 * here so they can be appended to the history exactly as received (thinking signatures included).
 */
public final class RelayClient {
    /** Accepted relay addresses: https, a host, nothing else (no query, no user info). */
    public static boolean validUrl(String u) {
        return u != null && u.matches("https://[A-Za-z0-9.-]+(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?");
    }

    public interface TextListener { void text(String delta); }

    public static final class RelayException extends IOException {
        public final int status;
        public RelayException(int status, String message) { super(status + ": " + message); this.status = status; }
    }

    public static final class Reply {
        public final List<Object> content = new ArrayList<>();
        public String stopReason;
        public String model;
        public final Map<String, Object> usage = new LinkedHashMap<>();

        public String text() {
            StringBuilder b = new StringBuilder();
            for (Object o : content) {
                Map<String, Object> m = MiniJson.obj(o);
                if ("text".equals(m.get("type"))) b.append((String) m.get("text"));
            }
            return b.toString();
        }
    }

    private final String url;
    private final String token;
    private volatile HttpURLConnection active;

    public RelayClient(String url, String token) {
        this.url = url.replaceAll("/+$", "");
        this.token = token;
    }

    /** Aborts a running {@link #chat} (the reader then throws). */
    public void cancel() {
        HttpURLConnection c = active;
        if (c != null) c.disconnect();
    }

    private HttpURLConnection open(String path, String body) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(url + path).openConnection();
        c.setConnectTimeout(15000);
        c.setReadTimeout(120000);
        c.setRequestProperty("Authorization", "Bearer " + token);
        c.setRequestProperty("Content-Type", "application/json");
        // Cloudflare rejects some default user agents with 403 / error 1010.
        c.setRequestProperty("User-Agent", "fereko-chat-android/0.4");
        if (body != null) {
            c.setRequestMethod("POST");
            c.setDoOutput(true);
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            c.setFixedLengthStreamingMode(bytes.length);
            try (OutputStream os = c.getOutputStream()) { os.write(bytes); }
        }
        int code = c.getResponseCode();
        if (code != 200) {
            String raw = readAll(c.getErrorStream());
            String msg = raw;
            try {
                Map<String, Object> e = MiniJson.obj(MiniJson.parse(raw));
                Object err = e.get("error");
                msg = err instanceof String ? (String) err : String.valueOf(MiniJson.obj(err).get("message"));
            } catch (RuntimeException ignored) { }
            c.disconnect();
            throw new RelayException(code, msg);
        }
        return c;
    }

    public Map<String, Object> usage() throws IOException {
        HttpURLConnection c = open("/v1/usage", null);
        try {
            return MiniJson.obj(MiniJson.parse(readAll(c.getInputStream())));
        } finally {
            c.disconnect();
        }
    }

    public Reply chat(String system, List<Object> messages, TextListener listener) throws IOException {
        return chat(system, messages, null, listener);
    }

    /** @param model "opus" / "sonnet" / "haiku", or null for the relay's default */
    public Reply chat(String system, List<Object> messages, String model, TextListener listener) throws IOException {
        HttpURLConnection c = open("/v1/chat", body(system, messages, model));
        active = c;
        Reply reply = new Reply();
        TreeMap<Integer, Map<String, Object>> blocks = new TreeMap<>();
        Map<Integer, StringBuilder> partialJson = new LinkedHashMap<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(c.getInputStream(), StandardCharsets.UTF_8))) {
            String event = null;
            StringBuilder data = new StringBuilder();
            String line;
            while ((line = r.readLine()) != null) {
                if (line.startsWith("event:")) {
                    event = line.substring(6).trim();
                } else if (line.startsWith("data:")) {
                    if (data.length() > 0) data.append('\n');
                    data.append(line.substring(5).replaceFirst("^ ", ""));
                } else if (line.isEmpty() && data.length() > 0) {
                    apply(event, MiniJson.obj(MiniJson.parse(data.toString())), reply, blocks, partialJson, listener);
                    event = null;
                    data.setLength(0);
                }
            }
        } finally {
            active = null;
            c.disconnect();
        }
        for (Map<String, Object> b : blocks.values()) {
            b.values().removeIf(v -> v == null);   // e.g. citations: null on a text block
            reply.content.add(b);
        }
        return reply;
    }

    /**
     * Writes the prompt cache ahead of the next turn (max_tokens 0, no reply), so the first answer
     * after opening the chat is as quick as the later ones. messages must end with a user turn.
     * @return the usage the relay reports
     */
    public Map<String, Object> warm(String system, List<Object> messages, String model) throws IOException {
        HttpURLConnection c = open("/v1/warm", body(system, messages, model));
        try {
            Object u = MiniJson.obj(MiniJson.parse(readAll(c.getInputStream()))).get("usage");
            return u == null ? new LinkedHashMap<>() : MiniJson.obj(u);
        } finally {
            c.disconnect();
        }
    }

    private static String body(String system, List<Object> messages, String model) {
        Map<String, Object> body = new LinkedHashMap<>();
        if (model != null) body.put("model", model);
        body.put("system", system);
        body.put("messages", messages);
        return MiniJson.write(body);
    }

    private static void apply(String event, Map<String, Object> d, Reply reply, TreeMap<Integer, Map<String, Object>> blocks,
                              Map<Integer, StringBuilder> partialJson, TextListener listener) throws RelayException {
        Object typeObj = d.get("type");
        String t = typeObj != null ? (String) typeObj : event;
        switch (t) {
            case "message_start": {
                Map<String, Object> m = MiniJson.obj(d.get("message"));
                reply.model = (String) m.get("model");
                if (m.get("usage") != null) reply.usage.putAll(MiniJson.obj(m.get("usage")));
                break;
            }
            case "content_block_start":
                blocks.put(index(d), new LinkedHashMap<>(MiniJson.obj(d.get("content_block"))));
                break;
            case "content_block_delta": {
                Map<String, Object> b = blocks.get(index(d));
                Map<String, Object> delta = MiniJson.obj(d.get("delta"));
                String kind = (String) delta.get("type");
                switch (kind) {
                    case "text_delta": {
                        String s = (String) delta.get("text");
                        b.put("text", str(b.get("text")) + s);
                        if (listener != null) listener.text(s);
                        break;
                    }
                    case "thinking_delta":
                        b.put("thinking", str(b.get("thinking")) + delta.get("thinking"));
                        break;
                    case "signature_delta":
                        b.put("signature", str(b.get("signature")) + delta.get("signature"));
                        break;
                    case "input_json_delta":
                        partialJson.computeIfAbsent(index(d), k -> new StringBuilder()).append((String) delta.get("partial_json"));
                        break;
                    case "citations_delta": {
                        Object cs = b.get("citations");
                        List<Object> list = cs instanceof List ? MiniJson.arr(cs) : new ArrayList<>();
                        list.add(delta.get("citation"));
                        b.put("citations", list);
                        break;
                    }
                    default:
                        throw new RelayException(0, "unknown delta type " + kind + "; the history would no longer match");
                }
                break;
            }
            case "content_block_stop": {
                StringBuilder pj = partialJson.remove(index(d));
                if (pj != null) blocks.get(index(d)).put("input", MiniJson.parse(pj.length() == 0 ? "{}" : pj.toString()));
                break;
            }
            case "message_delta": {
                Object dd = d.get("delta");
                if (dd != null && MiniJson.obj(dd).get("stop_reason") != null) reply.stopReason = (String) MiniJson.obj(dd).get("stop_reason");
                if (d.get("usage") != null) reply.usage.putAll(MiniJson.obj(d.get("usage")));
                break;
            }
            case "error": {
                Map<String, Object> e = MiniJson.obj(d.get("error"));
                throw new RelayException(0, e.get("type") + ": " + e.get("message"));
            }
            default:
                break;   // message_stop, ping
        }
    }

    private static int index(Map<String, Object> d) { return ((Double) d.get("index")).intValue(); }

    private static String str(Object o) { return o == null ? "" : (String) o; }

    private static String readAll(InputStream in) throws IOException {
        if (in == null) return "";
        try (InputStream s = in) { return new String(s.readAllBytes(), StandardCharsets.UTF_8); }
    }
}
