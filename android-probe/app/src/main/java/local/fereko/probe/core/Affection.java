package local.fereko.probe.core;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * ふぇれ子's affection (好感度) for キミ, 0..100 in five stages. Claude ends each reply with a hidden
 * tag ［好感度:+N］ (its own judgement of this exchange); the app clamps it (-5..+3 per reply, at
 * most +10 a day, so two smooth lines can't win her over) and sends the current value with the
 * next user message (not in the system prompt, so the prompt cache stays valid). The rules of
 * how she behaves at each stage are in persona.md.
 */
public final class Affection {
    public static final int MAX_UP = 3, MAX_DOWN = 5, DAILY_UP = 10;
    private static final String[] STAGES = {"他人行儀", "知り合い", "友だち", "気になる", "特別"};

    public static String stage(int v) { return STAGES[Math.min(4, Math.max(0, v) / 20)]; }

    /** The note added to the user's message: "〔好感度 12/100・他人行儀〕". */
    public static String note(int v) { return "〔好感度 " + v + "/100・" + stage(v) + "〕"; }

    private static final Pattern NOTE = Pattern.compile("\\n?〔好感度[^〕]*〕");
    private static final Pattern TAG = Pattern.compile("[［\\[]\\s*好感度\\s*[:：]\\s*([+＋\\-−－]?\\s*\\d+)\\s*[］\\]]");
    private static final Pattern ANY_TAG = Pattern.compile("\\s*[［\\[][^］\\]\\n]{1,20}[］\\]]");

    /** Text without the hidden note (user side) and tags (ふぇれ子's side), for showing and memos. */
    public static String clean(String s) {
        return ANY_TAG.matcher(NOTE.matcher(s).replaceAll("")).replaceAll("").trim();
    }

    /** The change Claude asked for in a finished reply, or null if it gave none. */
    public static Integer requested(String reply) {
        Matcher m = TAG.matcher(reply);
        Integer last = null;
        while (m.find()) {
            String n = m.group(1).replaceAll("\\s", "").replace('＋', '+').replace('−', '-').replace('－', '-');
            last = Integer.parseInt(n.startsWith("+") ? n.substring(1) : n);
        }
        return last;
    }

    /** The change actually applied: clamped per reply and by what is left of today's gain. */
    public static int allowed(int requested, int gainedToday) {
        if (requested < 0) return Math.max(-MAX_DOWN, requested);
        return Math.max(0, Math.min(Math.min(MAX_UP, requested), DAILY_UP - gainedToday));
    }

    /**
     * Streaming filter: removes ［...］ / [...] tags from Claude's text as it arrives, so they are
     * neither spoken nor shown. Text after an opening bracket is held until the bracket closes.
     */
    public static final class TagFilter {
        private final StringBuilder held = new StringBuilder();
        public final List<String> tags = java.util.Collections.synchronizedList(new ArrayList<>());

        public String feed(String delta) {
            StringBuilder out = new StringBuilder();
            for (int i = 0; i < delta.length(); i++) {
                char c = delta.charAt(i);
                if (held.length() > 0) {
                    held.append(c);
                    if (c == '］' || c == ']') {
                        tags.add(held.toString());
                        held.setLength(0);
                    } else if (held.length() > 24 || c == '\n') {   // not a tag after all
                        out.append(held);
                        held.setLength(0);
                    }
                } else if (c == '［' || c == '[') {
                    held.append(c);
                } else {
                    out.append(c);
                }
            }
            return out.toString();
        }

        /** End of the reply: an unclosed bracket was ordinary text. */
        public String flush() {
            String s = held.toString();
            held.setLength(0);
            return s;
        }
    }
}
