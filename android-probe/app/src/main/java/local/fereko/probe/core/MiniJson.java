package local.fereko.probe.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Minimal JSON reader (objects -> LinkedHashMap, arrays -> ArrayList, numbers -> Double). Same code on desktop and Android. */
public final class MiniJson {
    private final String s;
    private int i;

    private MiniJson(String s) { this.s = s; }

    public static Object parse(String text) {
        MiniJson p = new MiniJson(text);
        p.ws();
        Object v = p.value();
        p.ws();
        if (p.i != p.s.length()) throw p.err("trailing data");
        return v;
    }

    /** Serializes Map / List / String / Number / Boolean / null. Numbers that are whole are written without ".0". */
    public static String write(Object v) {
        StringBuilder b = new StringBuilder();
        write(b, v);
        return b.toString();
    }

    private static void write(StringBuilder b, Object v) {
        if (v == null) {
            b.append("null");
        } else if (v instanceof String) {
            quote(b, (String) v);
        } else if (v instanceof Boolean) {
            b.append(v);
        } else if (v instanceof Number) {
            double d = ((Number) v).doubleValue();
            if (d == Math.rint(d) && !Double.isInfinite(d) && Math.abs(d) < 1e15) b.append((long) d);
            else b.append(d);
        } else if (v instanceof Map) {
            b.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                if (!first) b.append(',');
                first = false;
                quote(b, String.valueOf(e.getKey()));
                b.append(':');
                write(b, e.getValue());
            }
            b.append('}');
        } else if (v instanceof List) {
            b.append('[');
            boolean first = true;
            for (Object o : (List<?>) v) {
                if (!first) b.append(',');
                first = false;
                write(b, o);
            }
            b.append(']');
        } else {
            throw new IllegalArgumentException("cannot serialize " + v.getClass());
        }
    }

    private static void quote(StringBuilder b, String s) {
        b.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        b.append('"');
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> obj(Object o) { return (Map<String, Object>) o; }

    @SuppressWarnings("unchecked")
    public static List<Object> arr(Object o) { return (List<Object>) o; }

    private Object value() {
        char c = s.charAt(i);
        switch (c) {
            case '{': return object();
            case '[': return array();
            case '"': return string();
            case 't': expect("true"); return Boolean.TRUE;
            case 'f': expect("false"); return Boolean.FALSE;
            case 'n': expect("null"); return null;
            default: return number();
        }
    }

    private Map<String, Object> object() {
        Map<String, Object> m = new LinkedHashMap<>();
        i++;
        ws();
        if (s.charAt(i) == '}') { i++; return m; }
        while (true) {
            ws();
            String k = string();
            ws();
            if (s.charAt(i++) != ':') throw err("expected ':'");
            ws();
            m.put(k, value());
            ws();
            char c = s.charAt(i++);
            if (c == '}') return m;
            if (c != ',') throw err("expected ',' or '}'");
        }
    }

    private List<Object> array() {
        List<Object> a = new ArrayList<>();
        i++;
        ws();
        if (s.charAt(i) == ']') { i++; return a; }
        while (true) {
            ws();
            a.add(value());
            ws();
            char c = s.charAt(i++);
            if (c == ']') return a;
            if (c != ',') throw err("expected ',' or ']'");
        }
    }

    private String string() {
        if (s.charAt(i++) != '"') throw err("expected string");
        StringBuilder b = new StringBuilder();
        while (true) {
            char c = s.charAt(i++);
            if (c == '"') return b.toString();
            if (c != '\\') { b.append(c); continue; }
            char e = s.charAt(i++);
            switch (e) {
                case '"': b.append('"'); break;
                case '\\': b.append('\\'); break;
                case '/': b.append('/'); break;
                case 'b': b.append('\b'); break;
                case 'f': b.append('\f'); break;
                case 'n': b.append('\n'); break;
                case 'r': b.append('\r'); break;
                case 't': b.append('\t'); break;
                case 'u': b.append((char) Integer.parseInt(s.substring(i, i + 4), 16)); i += 4; break;
                default: throw err("bad escape");
            }
        }
    }

    private Double number() {
        int st = i;
        while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
        if (st == i) throw err("unexpected character");
        return Double.parseDouble(s.substring(st, i));
    }

    private void expect(String w) {
        if (!s.startsWith(w, i)) throw err("expected " + w);
        i += w.length();
    }

    private void ws() {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
    }

    private IllegalArgumentException err(String m) {
        return new IllegalArgumentException("JSON: " + m + " at " + i);
    }
}
