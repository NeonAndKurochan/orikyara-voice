import local.fereko.probe.core.IrodoriEngine;
import local.fereko.probe.core.SelfTest;

import java.io.File;
import java.util.Map;

/** Runs the Android probe's core on the PC JVM: token fixtures, parity vs Python, and timings. */
public final class PcCheck {
    public static void main(String[] args) throws Exception {
        File pack = new File(args[0]);
        int threads = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        try (IrodoriEngine eng = new IrodoriEngine(pack, threads)) {
            System.out.println("load ms " + ms(eng.loadNanos));
            System.out.println(SelfTest.tokens(eng.tokenizer(), pack));
            SelfTest.Parity p = SelfTest.parity(eng, pack, null);
            System.out.println("parity " + (p.pass ? "PASS " : "FAIL ") + p.detail);
            for (int i = 0; i < 2; i++) {
                long t = System.nanoTime();
                IrodoriEngine.Result r = eng.synthesize("おはよう。今日は何をしようか。", null, 1, null);
                double sec = (System.nanoTime() - t) / 1e9, audio = r.audio.length / 48000.0;
                System.out.printf("run%d synth %.2fs audio %.2fs rtf %.2f stages %s%n", i, sec, audio, sec / audio, ms(r.stageNanos));
            }
        }
    }

    static String ms(Map<String, Long> m) {
        StringBuilder b = new StringBuilder("{");
        m.forEach((k, v) -> b.append(k).append('=').append(v / 1_000_000).append(' '));
        return b.append('}').toString();
    }
}
