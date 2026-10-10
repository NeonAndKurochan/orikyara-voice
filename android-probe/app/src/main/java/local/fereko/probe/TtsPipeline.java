package local.fereko.probe;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import local.fereko.probe.core.IrodoriEngine;

/**
 * Speaks a reply sentence by sentence while it is still streaming in. One worker synthesizes
 * sentences in order into a single StreamPlayer.
 *
 * Each sentence's audio is held back until the earliest time from which the rest of that
 * sentence can no longer run dry (estimated from its decode speed so far), then released.
 * So when synthesis is slower than playback, the shortfall becomes a pause *between* sentences
 * instead of stutter inside one.
 */
final class TtsPipeline {
    private static final String END = "\u0000end";
    private static final long START_MARGIN_NS = 150_000_000L;

    private final Future<IrodoriEngine> engineFuture;
    private final String caption;
    private final long seed;
    private final boolean useReference;
    private final LinkedBlockingQueue<String> queue = new LinkedBlockingQueue<>();
    private final ScheduledExecutorService timer = Executors.newSingleThreadScheduledExecutor();
    private final Thread worker;
    private volatile boolean cancelled;
    private volatile Probe.StreamPlayer player;

    final long t0;                       // System.nanoTime() when the user pressed send
    volatile long firstSentenceQueuedNs; // 0 until the first sentence is complete
    final List<Long> synthMs = new ArrayList<>();
    final List<Long> audioMs = new ArrayList<>();
    volatile String error;

    /** Audio of the sentence being synthesized that is not released to the player yet. */
    private final List<float[]> held = new ArrayList<>();
    private boolean released;
    private int sentenceNo;   // a scheduled release only applies to the sentence it was scheduled for

    TtsPipeline(Future<IrodoriEngine> engine, String caption, long seed, long t0, boolean useReference) {
        this.engineFuture = engine;
        this.useReference = useReference;
        this.caption = caption;
        this.seed = seed;
        this.t0 = t0;
        worker = new Thread(this::run, "tts");
        worker.start();
    }

    void say(String sentence) {
        if (firstSentenceQueuedNs == 0) firstSentenceQueuedNs = System.nanoTime();
        queue.add(sentence);
    }

    /** No more sentences: lets the worker finish and the player drain. */
    void end() {
        queue.add(END);
    }

    void cancel() {
        cancelled = true;
        queue.add(END);
        Probe.StreamPlayer p = player;
        if (p != null) p.stop();
    }

    void await() throws InterruptedException {
        worker.join();
    }

    long firstFrameNs() {
        Probe.StreamPlayer p = player;
        return p == null ? 0 : p.firstFrameNanos;
    }

    /** Loudness of ふぇれ子's voice right now (0..1), for the stand picture. */
    float level() {
        Probe.StreamPlayer p = player;
        return p == null || cancelled ? 0 : p.level();
    }

    int underruns() {
        Probe.StreamPlayer p = player;
        return p == null ? 0 : p.underruns;
    }

    private synchronized void release(int no) {
        if (released || no != sentenceNo) return;
        released = true;
        Probe.StreamPlayer p = player;
        for (float[] s : held) p.write(s);
        held.clear();
        if (!p.started) p.startAt(System.nanoTime());
    }

    private synchronized void add(float[] samples) {
        addedSamples += samples.length;
        if (keepAudio) synchronized (kept) { kept.add(samples.clone()); }
        if (released) player.write(samples);
        else held.add(samples);
    }

    /** For demo videos: keep every synthesized sample in playback order (set before the first say()). */
    volatile boolean keepAudio;
    private final List<float[]> kept = new ArrayList<>();

    /** The kept audio (48 kHz mono), all sentences back to back. */
    float[] keptAudio() {
        synchronized (kept) {
            int n = 0;
            for (float[] k : kept) n += k.length;
            float[] out = new float[n];
            int o = 0;
            for (float[] k : kept) { System.arraycopy(k, 0, out, o, k.length); o += k.length; }
            return out;
        }
    }

    /** {index of the sentence in say() order, first sample in the player stream}, in playback order */
    private final List<long[]> starts = new ArrayList<>();
    private long addedSamples;
    private int taken;

    /**
     * The sentence being heard now, as its index in say() order (sentences that turned out empty are
     * skipped but keep their index), or -1 before the first one.
     */
    int currentSentence() {
        Probe.StreamPlayer p = player;
        long head = p == null ? -1 : p.playedFrames();
        if (head < 0) return -1;
        int cur = -1;
        synchronized (starts) {
            for (long[] s : starts) if (s[1] <= head) cur = (int) s[0];
        }
        return cur;
    }

    private void run() {
        IrodoriEngine.Listener listener = new IrodoriEngine.Listener() {
            @Override public void stage(String name, long nanos) { }
            @Override public boolean cancelled() { return cancelled; }
        };
        long[] windowFrames = {0};
        try {
            IrodoriEngine engine = engineFuture.get();
            engine.setUseReference(useReference);
            while (!cancelled) {
                String s = queue.take();
                if (s == END) break;
                final int said = taken++;
                final boolean[] first = {true};
                final int no;
                synchronized (this) {
                    released = false;
                    held.clear();
                    no = ++sentenceNo;
                }
                windowFrames[0] = 0;
                long ts = System.nanoTime();
                IrodoriEngine.Result r;
                try {
                    r = engine.synthesize(s, null, seed, listener, (samples, idx, plan, decodeNs) -> {
                        if (player == null) player = new Probe.StreamPlayer(0, IrodoriEngine.SAMPLE_RATE);
                        if (first[0]) {
                            first[0] = false;
                            synchronized (starts) { starts.add(new long[]{said, addedSamples}); }
                        }
                        add(samples);
                        int[] c = plan.get(idx);
                        windowFrames[0] += c[3] - c[2];
                        if (idx == plan.size() - 1) {
                            release(no);
                            return;
                        }
                        long now = System.nanoTime();
                        long at = Probe.earliestStart(now, plan, idx, decodeNs / (double) windowFrames[0],
                                IrodoriEngine.HOP, IrodoriEngine.SAMPLE_RATE, START_MARGIN_NS);
                        // Released audio queues behind whatever is still playing, so the sentence
                        // effectively starts at max(at, end of the previous one): releasing at `at` is safe.
                        if (at <= now) release(no);
                        else timer.schedule(() -> release(no), at - now, TimeUnit.NANOSECONDS);
                    }, caption);
                } catch (IllegalArgumentException empty) {
                    continue;   // nothing speakable after normalization (e.g. only an emoji)
                }
                release(no);
                synchronized (this) {
                    synthMs.add((System.nanoTime() - ts) / 1_000_000);
                    audioMs.add(r.audio.length * 1000L / IrodoriEngine.SAMPLE_RATE);
                }
            }
        } catch (IrodoriEngine.CancelledException | InterruptedException ignored) {
        } catch (Exception e) {
            error = e.toString();
        } finally {
            timer.shutdownNow();
            Probe.StreamPlayer p = player;
            if (p != null) {
                if (cancelled) p.stop();
                p.finishAndWait();
            }
        }
    }
}
