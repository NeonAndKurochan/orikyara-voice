package local.fereko.probe;

import android.graphics.Typeface;
import android.text.Layout;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.AbsoluteSizeSpan;
import android.text.style.AlignmentSpan;
import android.text.style.ForegroundColorSpan;
import android.text.style.StyleSpan;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

/**
 * The conversation as shown over the picture: ふぇれ子's latest reply large and bright, older lines
 * smaller and dimmer, the user's lines small on the right, notes like 「（割り込み）」 small and grey
 * in the middle. No speech bubbles or name tags: who is talking is shown by size and side.
 * Takes the same strings the chat screen always wrote ("あなた：…", "ふぇれ子：" then deltas, "（…）").
 */
final class ChatLog {
    private enum Kind { USER, FERE, NOTE }

    private static final class Entry {
        final Kind kind;
        final StringBuilder text = new StringBuilder();
        Entry(Kind k, String t) { kind = k; text.append(t); }
    }

    static final int TEXT = 0xFFF4F4F6, DIM = 0xB3E8E8EC, USER = 0xFFC9C9D2, NOTE = 0xFF8E8E98;

    private final TextView view;
    private final float density;
    private final List<Entry> entries = new ArrayList<>();
    private Entry streaming;

    ChatLog(TextView view) {
        this.view = view;
        this.density = view.getResources().getDisplayMetrics().scaledDensity;
    }

    /** A whole line (main thread). */
    void add(String s) {
        String t = s.trim();
        if (t.isEmpty()) {          // the "\n" written after a reply: the reply is complete
            streaming = null;
            render();
            return;
        }
        if (t.startsWith("あなた：")) entries.add(new Entry(Kind.USER, t.substring(4).trim()));
        else if (t.startsWith("ふぇれ子：")) entries.add(new Entry(Kind.FERE, t.substring(5).trim()));
        else {
            streaming = null;
            entries.add(new Entry(Kind.NOTE, t));
        }
        render();
    }

    /** Streamed text (main thread): "ふぇれ子：" starts a reply, anything else continues it. */
    void addRaw(String s) {
        if (s.equals("ふぇれ子：")) {
            streaming = new Entry(Kind.FERE, "");
            entries.add(streaming);
        } else if (streaming != null) {
            streaming.text.append(s);
        } else {
            add(s);
            return;
        }
        render();
    }

    void clear() {
        entries.clear();
        streaming = null;
        render();
    }

    private void render() {
        int lastFere = -1;
        for (int i = entries.size() - 1; i >= 0; i--) {
            if (entries.get(i).kind == Kind.FERE) { lastFere = i; break; }
        }
        SpannableStringBuilder b = new SpannableStringBuilder();
        for (int i = 0; i < entries.size(); i++) {
            Entry e = entries.get(i);
            String t = e.text.toString().trim();
            if (t.isEmpty() && e != streaming) continue;
            if (b.length() > 0) b.append("\n\n");
            int start = b.length();
            b.append(t.isEmpty() ? "…" : t);
            int end = b.length();
            switch (e.kind) {
                case USER:
                    span(b, new AlignmentSpan.Standard(Layout.Alignment.ALIGN_OPPOSITE), start, end);
                    span(b, new ForegroundColorSpan(USER), start, end);
                    span(b, new AbsoluteSizeSpan(sp(14)), start, end);
                    break;
                case NOTE:
                    span(b, new AlignmentSpan.Standard(Layout.Alignment.ALIGN_CENTER), start, end);
                    span(b, new ForegroundColorSpan(NOTE), start, end);
                    span(b, new AbsoluteSizeSpan(sp(12)), start, end);
                    break;
                default:
                    boolean latest = i == lastFere;
                    span(b, new ForegroundColorSpan(latest ? TEXT : DIM), start, end);
                    span(b, new AbsoluteSizeSpan(sp(latest ? 19 : 15)), start, end);
                    if (latest) span(b, new StyleSpan(Typeface.BOLD), start, end);
                    break;
            }
        }
        view.setText(b);
    }

    private int sp(int v) { return Math.round(v * density); }

    private static void span(SpannableStringBuilder b, Object what, int start, int end) {
        b.setSpan(what, start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
    }
}
