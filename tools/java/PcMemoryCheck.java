import local.fereko.probe.core.ChatMemory;
import local.fereko.probe.core.MiniJson;
import local.fereko.probe.core.RelayClient;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

/**
 * Memory compaction on a real history (e.g. pulled from the phone), through the real relay with
 * Sonnet. Prints the cut, the transcript size and the resulting memo; writes the memo next to the
 * history as memory_test.md. The relay address and token are read from ~/.fereko/relay-url.txt / relay-token.txt and not printed.
 */
public final class PcMemoryCheck {
    public static void main(String[] args) throws Exception {
        File hist = new File(args[0]);
        List<Object> history = MiniJson.arr(MiniJson.parse(new String(Files.readAllBytes(hist.toPath()), StandardCharsets.UTF_8)));
        int cut = ChatMemory.cutIndex(history);
        System.out.println("messages " + history.size() + ", cut at " + cut);
        if (cut < 0) return;
        String transcript = ChatMemory.transcript(history.subList(0, cut));
        System.out.println("transcript chars " + transcript.length());
        String token = new String(Files.readAllBytes(new File(System.getProperty("user.home"), ".fereko/relay-token.txt").toPath()),
                StandardCharsets.UTF_8).trim();
        String url = new String(Files.readAllBytes(new File(System.getProperty("user.home"), ".fereko/relay-url.txt").toPath()),
                StandardCharsets.UTF_8).trim();
        RelayClient relay = new RelayClient(url, token);
        long t = System.nanoTime();
        RelayClient.Reply r = relay.chat(ChatMemory.SUMMARIZER, ChatMemory.requestMessages("", transcript), "sonnet", null);
        String memo = ChatMemory.cleanMemo(r.text());
        System.out.printf("summary %d ms, model %s, usage %s, memo chars %d%n", (System.nanoTime() - t) / 1_000_000,
                r.model, MiniJson.write(r.usage), memo.length());
        File out = new File(hist.getParentFile(), "memory_test.md");
        Files.write(out.toPath(), memo.getBytes(StandardCharsets.UTF_8));
        System.out.println("memo -> " + out);
    }
}
