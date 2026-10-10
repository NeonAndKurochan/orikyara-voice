package local.fereko.probe.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Cuts streamed reply text into sentences for TTS as soon as each one is complete.
 * Same rule as tools/fereko_chat.py: a sentence ends with 。！？!?… and keeps any emoji
 * or spaces that follow (the persona puts a style emoji at the end of a sentence).
 * A sentence is only released once the next non-emoji character arrives, so a trailing
 * emoji still in flight is not cut off, or once a line break follows it.
 */
public final class SentenceSplitter {
    private final StringBuilder buf = new StringBuilder();

    /** Feeds streamed text; returns sentences that are now complete. */
    public synchronized List<String> feed(String delta) {
        buf.append(delta);
        List<String> out = new ArrayList<>();
        while (true) {
            int end = completeEnd();
            if (end < 0) break;
            add(out, buf.substring(0, end));
            buf.delete(0, end);
        }
        return out;
    }

    /**
     * The stream has paused right after a sentence end: release that sentence without waiting for
     * the next character (Claude's text often comes in bursts ~0.8 s apart, so waiting delayed the
     * first sentence). An emoji that arrives later then starts the next sentence instead.
     */
    public synchronized List<String> releaseEnded() {
        List<String> out = new ArrayList<>();
        int n = buf.length(), last = n;
        while (last > 0 && isTrailer(buf.codePointBefore(last))) last -= Character.charCount(buf.codePointBefore(last));
        if (last > 0 && isEnd(buf.codePointBefore(last))) {
            out.addAll(feed(""));   // any complete sentences before it
            add(out, buf.toString());
            buf.setLength(0);
        }
        return out;
    }

    /** Returns whatever is left at the end of the reply. */
    public synchronized List<String> flush() {
        List<String> out = new ArrayList<>();
        add(out, buf.toString());
        buf.setLength(0);
        return out;
    }

    /** End index of the first complete sentence, or -1 if none is complete yet. */
    private int completeEnd() {
        int i = 0, n = buf.length();
        while (i < n && !isEnd(buf.codePointAt(i))) i += Character.charCount(buf.codePointAt(i));
        if (i >= n) return -1;
        while (i < n && isEnd(buf.codePointAt(i))) i += Character.charCount(buf.codePointAt(i));
        boolean newline = false;
        while (i < n && isTrailer(buf.codePointAt(i))) {
            newline |= buf.charAt(i) == '\n';
            i += Character.charCount(buf.codePointAt(i));
        }
        // need one more character to know the sentence is over, unless a line break already says so
        // (the persona puts one after each sentence, emoji before it; saves ~0.4-0.9 s on the first sentence)
        return i < n || newline ? i : -1;
    }

    private static void add(List<String> out, String s) {
        // an emoji that arrived after its sentence was already released (releaseEnded) is dropped
        int lead = 0;
        while (lead < s.length() && isTrailer(s.codePointAt(lead))) lead += Character.charCount(s.codePointAt(lead));
        s = TextNormalizer.pyStrip(s.substring(lead));
        if (s.isEmpty()) return;
        // glue very short fragments (e.g. a lone emoji) to the previous sentence
        if (!out.isEmpty() && s.codePointCount(0, s.length()) <= 2) out.set(out.size() - 1, out.get(out.size() - 1) + s);
        else out.add(s);
    }

    static boolean isEnd(int cp) {
        return cp == '。' || cp == '！' || cp == '？' || cp == '!' || cp == '?' || cp == '…';
    }

    static boolean isTrailer(int cp) {
        return Character.isWhitespace(cp) || (cp >= 0x1F300 && cp <= 0x1FAFF) || (cp >= 0x2600 && cp <= 0x27BF)
                || cp == 0xFE0F || cp == 0x200D;
    }
}
