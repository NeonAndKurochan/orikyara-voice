package local.fereko.probe.core;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Hugging Face `tokenizers` Unigram model as configured in the Irodori tokenizer.json:
 * no normalizer, Metaspace pre-tokenizer (replacement U+2581, prepend_scheme=never, split=false),
 * Viterbi over code points, unknown spans fused, byte_fallback to <0xXX> pieces.
 * Encodes without special tokens; the caller prepends BOS (irodori_tts/tokenizer.py).
 */
public final class UnigramTokenizer {
    private static final double UNK_PENALTY = 10.0;

    private final Map<String, Integer> ids = new HashMap<>();
    private final double[] scores;
    private final int unkId;
    private final boolean byteFallback;
    private final int maxPieceCodePoints;
    private final double unkScore;

    public UnigramTokenizer(String tokenizerJson) {
        Map<String, Object> root = MiniJson.obj(MiniJson.parse(tokenizerJson));
        Map<String, Object> model = MiniJson.obj(root.get("model"));
        if (!"Unigram".equals(model.get("type"))) throw new IllegalArgumentException("not a Unigram model");
        Object pre = root.get("pre_tokenizer");
        if (pre == null || !"Metaspace".equals(MiniJson.obj(pre).get("type"))
                || !"never".equals(MiniJson.obj(pre).get("prepend_scheme"))
                || !Boolean.FALSE.equals(MiniJson.obj(pre).get("split"))) {
            throw new IllegalArgumentException("unexpected pre_tokenizer: " + pre);
        }
        if (root.get("normalizer") != null) throw new IllegalArgumentException("unexpected normalizer");
        List<Object> vocab = MiniJson.arr(model.get("vocab"));
        scores = new double[vocab.size()];
        double min = Double.MAX_VALUE;
        int maxLen = 1;
        for (int i = 0; i < vocab.size(); i++) {
            List<Object> e = MiniJson.arr(vocab.get(i));
            String piece = (String) e.get(0);
            scores[i] = (Double) e.get(1);
            ids.putIfAbsent(piece, i);
            min = Math.min(min, scores[i]);
            maxLen = Math.max(maxLen, piece.codePointCount(0, piece.length()));
        }
        unkId = ((Double) model.get("unk_id")).intValue();
        byteFallback = Boolean.TRUE.equals(model.get("byte_fallback"));
        maxPieceCodePoints = maxLen;
        unkScore = min - UNK_PENALTY;
    }

    public int idOf(String piece) {
        Integer id = ids.get(piece);
        return id == null ? -1 : id;
    }

    /** tokenizer.encode(text, add_special_tokens=False).ids */
    public List<Integer> encode(String text) {
        String s = text.replace(" ", "▁");
        int[] cps = s.codePoints().toArray();
        int n = cps.length;
        // offsets[k] = UTF-16 index of code point k
        int[] off = new int[n + 1];
        for (int k = 0, u = 0; k < n; k++) { off[k] = u; u += Character.charCount(cps[k]); }
        off[n] = s.length();

        double[] best = new double[n + 1];
        int[] from = new int[n + 1];
        int[] tok = new int[n + 1];       // token id, or -1 for unknown
        boolean[] reached = new boolean[n + 1];
        reached[0] = true;
        for (int st = 0; st < n; st++) {
            if (!reached[st]) continue;
            boolean singleFound = false;
            int lim = Math.min(n, st + maxPieceCodePoints);
            for (int en = st + 1; en <= lim; en++) {
                Integer id = ids.get(s.substring(off[st], off[en]));
                if (id == null) continue;
                if (en == st + 1) singleFound = true;
                relax(best, from, tok, reached, st, en, id, scores[id]);
            }
            if (!singleFound) relax(best, from, tok, reached, st, st + 1, -1, unkScore);
        }

        // backtrack into (start, end, id) spans
        List<int[]> spans = new ArrayList<>();
        for (int en = n; en > 0; en = from[en]) spans.add(0, new int[]{from[en], en, tok[en]});

        // fuse consecutive unknown spans
        List<int[]> fused = new ArrayList<>();
        for (int[] sp : spans) {
            int[] last = fused.isEmpty() ? null : fused.get(fused.size() - 1);
            if (sp[2] == -1 && last != null && last[2] == -1) last[1] = sp[1];
            else fused.add(sp);
        }

        List<Integer> out = new ArrayList<>();
        for (int[] sp : fused) {
            if (sp[2] >= 0) { out.add(sp[2]); continue; }
            String piece = s.substring(off[sp[0]], off[sp[1]]);
            if (byteFallback) {
                byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
                List<Integer> b = new ArrayList<>();
                for (byte x : bytes) {
                    int id = idOf(String.format("<0x%02X>", x & 0xff));
                    if (id < 0) { b = null; break; }
                    b.add(id);
                }
                if (b != null) { out.addAll(b); continue; }
            }
            out.add(unkId);
        }
        return out;
    }

    private static void relax(double[] best, int[] from, int[] tok, boolean[] reached,
                              int st, int en, int id, double score) {
        double cand = best[st] + score;
        if (!reached[en] || cand > best[en]) {
            reached[en] = true;
            best[en] = cand;
            from[en] = st;
            tok[en] = id;
        }
    }

    /** BOS + body (truncated to maxLen - 1), no padding: matches PretrainedTextTokenizer.batch_encode up to the mask. */
    public long[] encodeWithBos(String normalized, int bosId, int maxLen) {
        List<Integer> body = encode(normalized);
        int len = Math.min(body.size(), maxLen - 1);
        long[] r = new long[len + 1];
        r[0] = bosId;
        for (int i = 0; i < len; i++) r[i + 1] = body.get(i);
        return r;
    }
}
