import local.fereko.probe.core.Fbank;
import local.fereko.probe.core.IrodoriEngine;
import local.fereko.probe.core.ReazonAsr;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/** Java ASR vs the Python reference (tools/asr_ref.py fixture): fbank values, token ids, timings. */
public final class PcAsrCheck {
    public static void main(String[] args) throws Exception {
        File model = new File(args[0]), fix = new File(args[1]);
        float[] audio = IrodoriEngine.readF32(new File(fix, "audio16k.f32"));
        float[] ref = IrodoriEngine.readF32(new File(fix, "fbank.f32"));
        String[] exp = new String(Files.readAllBytes(new File(fix, "expected.txt").toPath()), StandardCharsets.UTF_8).split("\n");

        float[] f = new Fbank().compute(audio, audio.length);
        double md = 0;
        for (int i = 0; i < Math.min(f.length, ref.length); i++) md = Math.max(md, Math.abs(f[i] - ref[i]));
        System.out.printf("fbank frames %d (py %s), maxdiff %.2e -> %s%n", f.length / Fbank.BINS, exp[0], md,
                f.length == ref.length && md < 1e-2 ? "OK" : "NG");

        try (ReazonAsr asr = new ReazonAsr(model, 4)) {
            System.out.printf("load %d ms%n", asr.loadNanos / 1_000_000);
            for (int i = 0; i < 2; i++) {
                ReazonAsr.Result r = asr.recognize(audio, audio.length);
                StringBuilder ids = new StringBuilder();
                for (int id : r.ids) ids.append(ids.length() == 0 ? "" : " ").append(id);
                System.out.printf("run%d audio %.1fs feat %d ms enc %d ms search %d ms ids %s%n", i, r.audioSec,
                        r.featNanos / 1_000_000, r.encNanos / 1_000_000, r.searchNanos / 1_000_000,
                        ids.toString().equals(exp[1].trim()) ? "OK" : "NG");
                Files.write(new File(fix, "java_text.txt").toPath(), r.text.getBytes(StandardCharsets.UTF_8));
            }
        }
    }
}
