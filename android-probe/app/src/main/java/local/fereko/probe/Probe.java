package local.fereko.probe;

import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTimestamp;
import android.media.AudioTrack;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Debug;
import android.os.PowerManager;

import org.json.JSONException;
import org.json.JSONObject;

/** Measurement helpers: memory sampling, thermal/battery state, playback with first-frame timing. */
final class Probe {
    private Probe() {}

    /** Samples this process's PSS every {@code intervalMs}; peaks between samples can be missed. */
    static final class MemorySampler implements AutoCloseable {
        final int intervalMs;
        volatile long peakPssKb;
        volatile long peakNativeKb;
        private volatile boolean running = true;
        private final Thread thread;

        MemorySampler(int intervalMs) {
            this.intervalMs = intervalMs;
            sample();
            thread = new Thread(() -> {
                while (running) {
                    sample();
                    try { Thread.sleep(intervalMs); } catch (InterruptedException e) { return; }
                }
            }, "pss-sampler");
            thread.start();
        }

        private void sample() {
            peakPssKb = Math.max(peakPssKb, Debug.getPss());
            peakNativeKb = Math.max(peakNativeKb, Debug.getNativeHeapAllocatedSize() / 1024);
        }

        @Override
        public void close() {
            sample();
            running = false;
            thread.interrupt();
        }
    }

    static JSONObject thermal(Context ctx) throws JSONException {
        JSONObject o = new JSONObject();
        PowerManager pm = ctx.getSystemService(PowerManager.class);
        o.put("thermal_status", pm.getCurrentThermalStatus());
        float headroom = pm.getThermalHeadroom(0);
        o.put("thermal_headroom", Float.isNaN(headroom) ? JSONObject.NULL : (double) headroom);
        Intent b = ctx.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (b != null) {
            o.put("battery_temp_c", b.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0);
            o.put("battery_level", b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1));
            o.put("plugged", b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0);
        }
        return o;
    }

    static JSONObject device(Context ctx) throws JSONException {
        ActivityManager am = ctx.getSystemService(ActivityManager.class);
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        am.getMemoryInfo(mi);
        JSONObject o = new JSONObject();
        o.put("manufacturer", Build.MANUFACTURER);
        o.put("model", Build.MODEL);
        o.put("soc", Build.SOC_MODEL);
        o.put("android", Build.VERSION.RELEASE);
        o.put("sdk", Build.VERSION.SDK_INT);
        o.put("ram_total_bytes", mi.totalMem);
        o.put("ram_avail_bytes", mi.availMem);
        o.put("cpus", Runtime.getRuntime().availableProcessors());
        return o;
    }

    /**
     * Streams mono float PCM while it is still being decoded. Audio is written as chunks arrive;
     * playback starts at {@link #startAt} (moved earlier or later by the caller as estimates improve).
     */
    static final class StreamPlayer {
        private static final float[] END = new float[0];
        private final AudioTrack track;
        private final int sampleRate;
        private final Thread starter;
        private final Thread writer;
        private final java.util.concurrent.LinkedBlockingQueue<float[]> queue = new java.util.concurrent.LinkedBlockingQueue<>();
        private volatile long startAtNanos = Long.MAX_VALUE;
        private volatile boolean stopped;
        volatile boolean started;
        volatile long playCallNanos;
        volatile long firstFrameNanos;   // 0 if no timestamp was obtained
        private volatile int written;
        int underruns;
        /** RMS of each 20 ms block in the order written, for level() (the stand picture's motion). */
        private final java.util.List<Float> envelope = java.util.Collections.synchronizedList(new java.util.ArrayList<>());
        private double envSum;
        private int envN;

        StreamPlayer(int capacitySamples, int sampleRate) {
            this.sampleRate = sampleRate;
            int minBuf = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT);
            track = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    // Small buffer on purpose: a streaming track only starts once its buffer is full,
                    // so a buffer sized for the whole utterance would not play until decoding ends.
                    .setBufferSizeInBytes(Math.max(minBuf, sampleRate / 4 * 4))
                    .build();
            starter = new Thread(this::waitAndStart, "stream-start");
            starter.start();
            writer = new Thread(this::writeLoop, "stream-write");
            writer.start();
        }

        /** Queues decoded audio; a writer thread feeds the track as it drains. */
        void write(float[] s) {
            int block = sampleRate / 50;
            for (float v : s) {
                envSum += v * v;
                if (++envN == block) {
                    envelope.add((float) Math.sqrt(envSum / block));
                    envSum = 0;
                    envN = 0;
                }
            }
            queue.add(s);
        }

        /** Frames heard so far (playback head), or -1 before the start / after stopping. */
        long playedFrames() {
            if (!started || stopped) return -1;
            try { return track.getPlaybackHeadPosition() & 0xffffffffL; } catch (IllegalStateException e) { return -1; }
        }

        /** Level (RMS, 0..1) of the audio being heard right now; 0 before start or after the end. */
        float level() {
            if (!started || stopped) return 0;
            int i;
            try { i = track.getPlaybackHeadPosition() / (sampleRate / 50); } catch (IllegalStateException e) { return 0; }
            return i >= 0 && i < envelope.size() ? envelope.get(i) : 0;
        }

        private void writeLoop() {
            try {
                while (!stopped) {
                    float[] s = queue.take();
                    if (s == END) return;
                    int off = 0;
                    while (!stopped && off < s.length) {
                        int n = track.write(s, off, s.length - off, AudioTrack.WRITE_NON_BLOCKING);
                        if (n < 0) return;
                        off += n;
                        written += n;
                        if (off < s.length) Thread.sleep(5);
                    }
                }
            } catch (InterruptedException ignored) { }
        }

        /** Playback begins at this System.nanoTime() value (or immediately if it has passed). */
        void startAt(long nanos) {
            startAtNanos = nanos;
            starter.interrupt();
        }

        private void waitAndStart() {
            while (!stopped) {
                long wait = startAtNanos - System.nanoTime();
                if (wait <= 0) break;
                try { Thread.sleep(Math.min(wait / 1_000_000 + 1, 50)); } catch (InterruptedException ignored) { }
            }
            if (stopped) return;
            playCallNanos = System.nanoTime();
            track.play();
            started = true;
            AudioTimestamp ts = new AudioTimestamp();
            long deadline = playCallNanos + 2_000_000_000L;
            while (!stopped && System.nanoTime() < deadline) {
                if (track.getTimestamp(ts) && ts.framePosition > 0) {
                    firstFrameNanos = ts.nanoTime - ts.framePosition * 1_000_000_000L / sampleRate;
                    return;
                }
                try { Thread.sleep(2); } catch (InterruptedException ignored) { }
            }
        }

        /** Call after the last write: makes sure playback starts, waits for the end, releases. */
        void finishAndWait() {
            queue.add(END);
            if (startAtNanos == Long.MAX_VALUE) startAt(System.nanoTime());
            try { starter.join(); writer.join(); } catch (InterruptedException ignored) { }
            while (!stopped && track.getPlaybackHeadPosition() < written) {
                try { Thread.sleep(20); } catch (InterruptedException e) { break; }
            }
            underruns = track.getUnderrunCount();
            try { track.stop(); } catch (IllegalStateException ignored) { }
            track.release();
        }

        void stop() {
            stopped = true;
            starter.interrupt();
            writer.interrupt();
        }
    }

    /**
     * Earliest System.nanoTime() at which playback can start without running dry, given that chunk
     * {@code doneIdx} just finished and later chunks decode at {@code nsPerWindowFrame}.
     * Each later chunk must be ready before its first sample is due: ready_j - start_j/rate.
     */
    static long earliestStart(long now, java.util.List<int[]> plan, int doneIdx, double nsPerWindowFrame,
                              int hop, int sampleRate, long marginNs) {
        long t = now;
        long need = now;
        for (int j = doneIdx + 1; j < plan.size(); j++) {
            int[] c = plan.get(j);
            t += (long) ((c[3] - c[2]) * nsPerWindowFrame);
            long due = (long) (c[0] * (double) hop * 1e9 / sampleRate);
            need = Math.max(need, t - due + marginNs);
        }
        return need;
    }

    /** Plays mono float PCM; {@link #firstFrameNanos} is frame 0's presentation time from AudioTimestamp. */
    static final class Player {
        private final AudioTrack track;
        private final int frames;
        long playCallNanos;
        long firstFrameNanos;   // 0 if no timestamp was obtained
        private volatile boolean stopped;

        Player(float[] audio, int sampleRate) {
            frames = audio.length;
            int minBuf = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT);
            track = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .setBufferSizeInBytes(Math.max(minBuf, audio.length * 4))
                    .build();
            track.write(audio, 0, audio.length, AudioTrack.WRITE_BLOCKING);
        }

        /** Starts playback and waits (up to 2 s) for the first timestamp. */
        void start(int sampleRate) {
            playCallNanos = System.nanoTime();
            track.play();
            AudioTimestamp ts = new AudioTimestamp();
            long deadline = playCallNanos + 2_000_000_000L;
            while (!stopped && System.nanoTime() < deadline) {
                if (track.getTimestamp(ts) && ts.framePosition > 0) {
                    firstFrameNanos = ts.nanoTime - ts.framePosition * 1_000_000_000L / sampleRate;
                    return;
                }
                try { Thread.sleep(2); } catch (InterruptedException e) { return; }
            }
        }

        void waitDone() {
            while (!stopped && track.getPlaybackHeadPosition() < frames) {
                try { Thread.sleep(20); } catch (InterruptedException e) { break; }
            }
            release();
        }

        void stop() {
            stopped = true;
        }

        private void release() {
            try { track.stop(); } catch (IllegalStateException ignored) { }
            track.release();
        }
    }
}
