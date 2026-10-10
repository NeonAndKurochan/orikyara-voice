package local.fereko.probe.core;

import java.text.Normalizer;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

/** Port of irodori_tts/text_normalization.py @ 89f9d8f (MIT). Order of steps matters. */
public final class TextNormalizer {
    private TextNormalizer() {}

    private static final Map<String, String> SIMPLE = new LinkedHashMap<>();
    static {
        SIMPLE.put("\t", "");
        SIMPLE.put("[n]", "");
        SIMPLE.put("\\[n\\]", "");   // literal backslashes, as in the Python raw string
        SIMPLE.put("　", "");
        SIMPLE.put("？", "?");
        SIMPLE.put("！", "!");
        SIMPLE.put("♥", "♡");
        SIMPLE.put("●", "○");
        SIMPLE.put("◯", "○");
        SIMPLE.put("〇", "○");
    }

    private static final Pattern[] REGEX = {
        Pattern.compile("[;▼♀♂《》≪≫①②③④⑤⑥]"),
        Pattern.compile("[\\u02d7\\u2010-\\u2015\\u2043\\u2212\\u23af\\u23e4\\u2500\\u2501\\u2e3a\\u2e3b]"),
        Pattern.compile("[\\uff5e\\u301C]"),
        Pattern.compile("…{3,}"),
    };
    private static final String[] REGEX_REP = {"", "", "ー", "……"};

    public static String stripOuterBrackets(String text) {
        String open = "「『（【(";
        String close = "」』）】)";
        while (text.length() >= 2) {
            char s = text.charAt(0);
            char e = text.charAt(text.length() - 1);
            int k = open.indexOf(s);
            if (k >= 0 && close.charAt(k) == e) {
                int depth = 0;
                boolean enclosing = true;
                for (int i = 0; i < text.length(); i++) {
                    char c = text.charAt(i);
                    if (c == s) depth++;
                    else if (c == e) depth--;
                    if (depth == 0 && i < text.length() - 1) { enclosing = false; break; }
                }
                if (enclosing && depth == 0) { text = text.substring(1, text.length() - 1); continue; }
            }
            break;
        }
        return text;
    }

    /** normalize_text(text).strip() as used by inference_runtime.synthesize. */
    public static String normalize(String text) {
        for (Map.Entry<String, String> e : SIMPLE.entrySet()) text = text.replace(e.getKey(), e.getValue());
        for (int i = 0; i < REGEX.length; i++) text = REGEX[i].matcher(text).replaceAll(REGEX_REP[i]);
        text = stripOuterBrackets(text);
        text = Normalizer.normalize(text, Normalizer.Form.NFKC);
        text = text.replace("...", "…").replace("..", "…");
        return pyStrip(text);
    }

    /** Python str.strip(): removes leading/trailing characters where str.isspace() is true. */
    static String pyStrip(String t) {
        int a = 0, b = t.length();
        while (a < b && isPySpace(t.charAt(a))) a++;
        while (b > a && isPySpace(t.charAt(b - 1))) b--;
        return t.substring(a, b);
    }

    private static boolean isPySpace(char c) {
        return Character.isWhitespace(c) || Character.isSpaceChar(c) || (c >= 0x1c && c <= 0x1f) || c == 0x85;
    }
}
