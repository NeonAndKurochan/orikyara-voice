package local.fereko.probe.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Long-term memory for the chat. Sending the whole history every turn makes each reply slower and
 * more expensive as the conversation grows, so once it passes COMPACT_AT messages the older part is
 * folded into a short memo ("what ふぇれ子 remembers": about キミ, running jokes, recent events) and
 * only the last ~KEEP messages are sent verbatim. The memo goes into the system prompt.
 */
public final class ChatMemory {
    private ChatMemory() {}

    public static final int COMPACT_AT = 48, KEEP = 24;
    /** Hard cap on the memo, so the system prompt stays well under the relay's 8000-character limit. */
    public static final int MAX_MEMO_CHARS = 2500;

    /** System prompt for the summarizing call (not ふぇれ子 herself). */
    public static final String SUMMARIZER =
            "あなたは会話アプリの「記憶係」。キャラクター「ふぇれ子」とユーザー（ふぇれ子は「キミ」と呼ぶ）の会話から、"
            + "次に話すときにふぇれ子が覚えておくべきことを、これまでのメモと合わせて書き直す。\n\n"
            + "書き方：\n"
            + "- 日本語の短い箇条書き（「- 」で始める）。見出しは「## キミのこと」「## 二人の話題・お約束」「## 最近の出来事」の3つ\n"
            + "- キミのこと：名前や呼び方、好きなもの、苦手なもの、仕事や生活、体調など、会話から分かった事実\n"
            + "- 二人の話題・お約束：繰り返し出たネタ、ツッコミのお約束、ふぇれ子について決まったこと（一人称など）\n"
            + "- 最近の出来事：キミに起きたこと、これからの予定や約束。古くなったものは消してよい\n"
            + "- 会話にないことは書かない。推測は「たぶん」と書く。あいさつや中身のないやりとりは書かない\n"
            + "- これまでのメモの内容は、新しい会話で変わったところ以外は残す\n"
            + "- 全体で1500文字以内。メモだけを出力して、前置きや説明は書かない";

    /**
     * Where the verbatim part should start: about KEEP messages from the end, moved forward to a
     * user turn (the request must start with one). -1 when no compaction is needed yet.
     */
    public static int cutIndex(List<Object> history) {
        if (history.size() <= COMPACT_AT) return -1;
        int i = history.size() - KEEP;
        while (i < history.size() && !isPlainUser(MiniJson.obj(history.get(i)))) i++;
        return i > 0 && i < history.size() ? i : -1;
    }

    private static boolean isPlainUser(Map<String, Object> m) {
        return "user".equals(m.get("role")) && m.get("content") instanceof String;
    }

    /** The messages as plain lines for the summarizer ("キミ：…" / "ふぇれ子：…", text only). */
    public static String transcript(List<Object> msgs) { return transcript(msgs, "ふぇれ子"); }

    public static String transcript(List<Object> msgs, String name) {
        StringBuilder b = new StringBuilder();
        for (Object o : msgs) {
            Map<String, Object> m = MiniJson.obj(o);
            String text = Affection.clean(text(m.get("content")));
            if (text.isEmpty()) continue;
            if (text.startsWith(EVENT_OPEN)) {   // the app was opened: keep it short in the memo input
                b.append("（アプリを開いた）\n");
                continue;
            }
            b.append("user".equals(m.get("role")) ? "キミ：" : name + "：").append(text.replace("\n", " ")).append('\n');
        }
        return b.toString();
    }

    public static String text(Object content) {
        if (content instanceof String) return (String) content;
        StringBuilder b = new StringBuilder();
        for (Object o : MiniJson.arr(content)) {
            Map<String, Object> blk = MiniJson.obj(o);
            if ("text".equals(blk.get("type"))) b.append((String) blk.get("text"));
        }
        return b.toString();
    }

    /** The user message for the summarizer. */
    public static String request(String memo, String transcript) {
        return "これまでのメモ：\n" + (memo == null || memo.trim().isEmpty() ? "（まだない）" : memo.trim())
                + "\n\n新しく古くなった会話：\n" + transcript + "\n更新したメモだけを出力して。";
    }

    public static List<Object> requestMessages(String memo, String transcript) {
        List<Object> msgs = new ArrayList<>();
        Map<String, Object> m = new java.util.LinkedHashMap<>();
        m.put("role", "user");
        m.put("content", request(memo, transcript));
        msgs.add(m);
        return msgs;
    }

    /** The summarizer prompt for another character (SUMMARIZER is written for ふぇれ子). */
    public static String summarizer(String name) { return SUMMARIZER.replace("ふぇれ子", name); }

    /** Start of the event message sent when the app is opened (not the user's words). */
    public static final String EVENT_OPEN = "（アプリが開かれた。";

    /**
     * The "app was opened" event: time of day and how long since they last talked, asking for a
     * short first line from ふぇれ子.
     * @param sinceMs time since the last turn, or -1 if they have never talked
     */
    public static String openEvent(java.util.Date now, long sinceMs) { return openEvent(now, sinceMs, "ふぇれ子"); }

    public static String openEvent(java.util.Date now, long sinceMs, String name) {
        java.util.Calendar c = java.util.Calendar.getInstance();
        c.setTime(now);
        int h = c.get(java.util.Calendar.HOUR_OF_DAY), m = c.get(java.util.Calendar.MINUTE);
        String part = h < 4 ? "深夜" : h < 10 ? "朝" : h < 16 ? "昼" : h < 19 ? "夕方" : "夜";
        String gap;
        if (sinceMs < 0) gap = "キミと話すのは初めて";
        else if (sinceMs < 2 * 3600_000L) gap = "前に話してから" + Math.max(1, sinceMs / 60_000) + "分くらい";
        else if (sinceMs < 2 * 86_400_000L) gap = "前に話してから" + sinceMs / 3600_000 + "時間くらい";
        else gap = "前に話してから" + sinceMs / 86_400_000 + "日ぶり";
        return EVENT_OPEN + "今は" + part + "の" + h + "時" + m + "分。" + gap
                + "。キミはまだ何も言っていない。" + name + "から短く一言声をかける）";
    }

    /** Clean up the summarizer's answer: trimmed and capped. */
    public static String cleanMemo(String s) {
        s = s == null ? "" : s.trim();
        if (s.length() > MAX_MEMO_CHARS) {
            int cut = s.lastIndexOf('\n', MAX_MEMO_CHARS);
            s = s.substring(0, cut > 0 ? cut : MAX_MEMO_CHARS).trim();
        }
        return s;
    }

    /** ふぇれ子's system prompt: the persona, what she remembers, and today's date. */
    public static String systemPrompt(String persona, String memo, String today) {
        StringBuilder b = new StringBuilder(persona.trim());
        if (memo != null && !memo.trim().isEmpty()) {
            b.append("\n\n## 覚えていること（前の会話のメモ。自然に活かす。メモを読み上げたりはしない）\n").append(memo.trim());
        }
        if (today != null && !today.isEmpty()) b.append("\n\n## 今日\n").append(today);
        return b.toString();
    }
}
