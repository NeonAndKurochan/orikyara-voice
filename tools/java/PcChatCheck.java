import local.fereko.probe.core.IrodoriEngine;
import local.fereko.probe.core.MiniJson;
import local.fereko.probe.core.RelayClient;
import local.fereko.probe.core.SentenceSplitter;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * PC checks for the v0.4 chat pieces:
 *   caption parity with Python (out/caption_golden), sentence splitting of streamed text,
 *   and RelayClient against a relay URL (use the local wrangler dev + mock, never the real one here).
 */
public final class PcChatCheck {
    public static void main(String[] args) throws Exception {
        File pack = new File(args[0]);
        File golden = new File(args[1]);
        String relayUrl = args.length > 2 ? args[2] : null;
        String relayToken = args.length > 3 ? args[3] : null;

        // 1. caption parity
        Map<String, Object> meta = MiniJson.obj(MiniJson.parse(IrodoriEngine.read(new File(golden, "meta.json"))));
        try (IrodoriEngine eng = new IrodoriEngine(pack, 6)) {
            float[] noise = IrodoriEngine.readF32(new File(golden, "noise.f32"));
            float[] exp = IrodoriEngine.readF32(new File(golden, "latent.f32"));
            IrodoriEngine.Result r = eng.synthesize((String) meta.get("text"), noise, 0, null, null, (String) meta.get("caption"));
            double md = 0;
            for (int i = 0; i < exp.length; i++) md = Math.max(md, Math.abs(exp[i] - r.latent[i]));
            System.out.printf("caption parity: frames %d (py %s), latent maxdiff %.2e -> %s%n",
                    r.latentFrames, meta.get("frames"), md, r.latent.length == exp.length && md < 1e-3 ? "OK" : "NG");
        }

        // 2. streamed sentence splitting
        String reply = "えー、ほんとに？すごいじゃん！😆 それで、そのあとどうなったの…？気になる🤭";
        SentenceSplitter sp = new SentenceSplitter();
        List<String> got = new ArrayList<>();
        for (int i = 0; i < reply.length(); ) {   // feed 1-3 chars at a time, as a stream would
            int n = Math.min(reply.length() - i, 1 + (i % 3));
            if (Character.isHighSurrogate(reply.charAt(i + n - 1)) && i + n < reply.length()) n++;
            for (String s : sp.feed(reply.substring(i, i + n))) got.add(s + "  <-released at char " + (i + n));
            i += n;
        }
        for (String s : sp.flush()) got.add(s + "  <-flush");
        got.forEach(s -> System.out.println("sentence: " + s));

        // 3. relay client
        if (relayUrl != null) {
            RelayClient rc = new RelayClient(relayUrl, relayToken);
            System.out.println("usage: " + rc.usage());
            List<Object> msgs = new ArrayList<>();
            msgs.add(MiniJson.parse("{\"role\":\"user\",\"content\":\"元気？\"}"));
            StringBuilder streamed = new StringBuilder();
            RelayClient.Reply rep = rc.chat("あなたはふぇれ子", msgs, streamed::append);
            System.out.println("streamed: " + streamed + " | stop " + rep.stopReason + " | model " + rep.model);
            System.out.println("content: " + MiniJson.write(rep.content));
            System.out.println("usage: " + MiniJson.write(rep.usage));
            try {
                new RelayClient(relayUrl, "wrong").usage();
            } catch (RelayClient.RelayException e) {
                System.out.println("wrong token -> " + e.status);
            }
        }
    }
}
