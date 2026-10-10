package local.fereko.probe.core;

import java.io.File;
import java.util.List;
import java.util.Map;

/** Checks against fixtures produced on PC by tools/prep_phone_pack.py. */
public final class SelfTest {
    private SelfTest() {}

    /** @return "tokens: N/N ok" plus the first few mismatches. */
    public static String tokens(UnigramTokenizer tok, File packDir) throws Exception {
        List<Object> fx = MiniJson.arr(MiniJson.parse(IrodoriEngine.read(new File(packDir, "fixtures/tokens.json"))));
        int ok = 0;
        StringBuilder bad = new StringBuilder();
        for (Object o : fx) {
            Map<String, Object> f = MiniJson.obj(o);
            String norm = TextNormalizer.normalize((String) f.get("text"));
            long[] ids = tok.encodeWithBos(norm, 1, IrodoriEngine.TEXT_MAX);
            List<Object> exp = MiniJson.arr(f.get("ids"));
            boolean same = norm.equals(f.get("normalized")) && ids.length == exp.size();
            for (int i = 0; same && i < ids.length; i++) same = ids[i] == ((Double) exp.get(i)).longValue();
            if (same) ok++;
            else if (bad.length() < 600) {
                bad.append("\n  NG ").append(f.get("text")).append(" -> ").append(norm)
                   .append(" ").append(java.util.Arrays.toString(ids)).append(" expected ").append(exp);
            }
        }
        return "tokens: " + ok + "/" + fx.size() + " ok" + bad;
    }

    public static final class Parity {
        public boolean pass;
        public double latentMaxDiff, audioMaxDiff;
        public String detail;
    }

    /** Same text and same initial noise as the PC golden run; compares latent and waveform. */
    public static Parity parity(IrodoriEngine eng, File packDir, IrodoriEngine.Listener l) throws Exception {
        Map<String, Object> meta = MiniJson.obj(MiniJson.parse(IrodoriEngine.read(new File(packDir, "fixtures/parity.json"))));
        float[] noise = IrodoriEngine.readF32(new File(packDir, "fixtures/parity_noise.f32"));
        float[] expLatent = IrodoriEngine.readF32(new File(packDir, "fixtures/parity_latent.f32"));
        float[] expAudio = IrodoriEngine.readF32(new File(packDir, "fixtures/parity_audio.f32"));
        // The golden run has no reference voice: compare without the registered one.
        eng.setUseReference(false);
        IrodoriEngine.Result r;
        try {
            r = eng.synthesize((String) meta.get("text"), noise, 0, l);
        } finally {
            eng.setUseReference(true);
        }
        Parity p = new Parity();
        p.latentMaxDiff = maxDiff(r.latent, expLatent);
        p.audioMaxDiff = maxDiff(r.audio, expAudio);
        boolean sameLen = r.audio.length == expAudio.length && r.latent.length == expLatent.length;
        // Different CPUs / thread counts reorder float sums; allow small drift, not structural differences.
        p.pass = sameLen && p.latentMaxDiff < 1e-2 && p.audioMaxDiff < 1e-2;
        p.detail = String.format("frames %d (pc %s), samples %d (pc %d), latent maxdiff %.2e, audio maxdiff %.2e",
                r.latentFrames, meta.get("latent_frames"), r.audio.length, expAudio.length, p.latentMaxDiff, p.audioMaxDiff);
        return p;
    }

    static double maxDiff(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        double m = a.length == b.length ? 0 : Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
        return m;
    }
}
