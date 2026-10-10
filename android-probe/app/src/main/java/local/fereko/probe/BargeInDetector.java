package local.fereko.probe;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.NoiseSuppressor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BooleanSupplier;

/**
 * Microphone with a simple level-based voice detector, for our own speech recognizer.
 *
 * BARGE_IN: detects the user starting to talk while ふぇれ子 is speaking. Input is the
 * voice-communication source with the platform echo canceller (on the Nothing Phone (3) it removes
 * ふぇれ子's own voice almost completely: echo measured at -91..-120 dBFS). Once playback has started,
 * the echo level is measured for CALIBRATE_MS; the trigger is MARGIN_DB above it (and above the
 * floor) for TRIGGER_MS. LISTEN: the same for a normal turn (🎤 / hands-free), calibrated on room
 * noise, with a lower floor, giving up after WAIT_MS of nothing.
 *
 * After the trigger the audio is kept (with the PREROLL_MS before it, so the first words are not
 * lost) until END_SILENCE_MS of quiet or MAX_UTTERANCE_MS, then the microphone is released and the
 * whole utterance goes to Listener.utterance().
 */
final class BargeInDetector {
    static final int RATE = 16000;
    private static final int FRAME = RATE / 50;          // 20 ms
    private static final double MARGIN_DB = 10;
    private static final int TRIGGER_MS = 160;
    private static final int TRIGGER_WINDOW_MS = 300;
    private static final int PREROLL_MS = 800;
    private static final int END_SILENCE_MS = 900;
    /**
     * While capturing, "still talking" uses a lower level than the trigger: ふぇれ子 has stopped (no
     * echo left), and quiet syllables at -38..-45 dBFS used to count as silence and cut the
     * utterance short (0.9.1: 「聴いてみたらもう」).
     */
    private static final double CONTINUE_FLOOR_DB = -52;
    private static final int MAX_UTTERANCE_MS = 15000;
    private static final int WAIT_MS = 8000;

    enum Mode {
        BARGE_IN(600, 0.9, -38), LISTEN(300, 0.1, -45);
        final int calibrateMs;
        final double percentile, floorDb;
        Mode(int calibrateMs, double percentile, double floorDb) {
            this.calibrateMs = calibrateMs;
            this.percentile = percentile;
            this.floorDb = floorDb;
        }
    }

    interface Listener {
        /** The user started talking (in BARGE_IN mode: stop playback now). */
        void bargeIn();
        /**
         * The utterance once the user stopped talking: 16 kHz mono in [-1, 1], microphone already
         * released. n == 0 if nothing was said (LISTEN timeout, or finish() before speech).
         * Not called after abort().
         */
        void utterance(float[] pcm, int n);
    }

    private final Thread thread;
    private final Mode mode;
    private volatile boolean running = true, aborted, finishNow;
    volatile boolean fired;
    volatile double echoDb = Double.NaN, thresholdDb = Double.NaN, continueDb = Double.NaN, maxDb = -120;
    volatile boolean aec;
    volatile String error;
    volatile int utteranceMs;
    /** nanoTime when the end of speech was decided */
    volatile long endNs;

    /** @param playing true once ふぇれ子's audio is actually playing (BARGE_IN calibration waits for it) */
    BargeInDetector(Mode mode, BooleanSupplier playing, Listener listener) {
        this.mode = mode;
        thread = new Thread(() -> run(playing, listener), mode == Mode.LISTEN ? "mic-listen" : "barge-in");
        thread.start();
    }

    @SuppressWarnings("MissingPermission")   // the caller checks RECORD_AUDIO
    private void run(BooleanSupplier playing, Listener listener) {
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord rec = null;
        AcousticEchoCanceler ec = null;
        NoiseSuppressor ns = null;
        ArrayDeque<short[]> preroll = new ArrayDeque<>();
        List<short[]> utt = new ArrayList<>();
        try {
            rec = new AudioRecord(MediaRecorder.AudioSource.VOICE_COMMUNICATION, RATE, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, Math.max(min, FRAME * 8));
            if (AcousticEchoCanceler.isAvailable()) {
                ec = AcousticEchoCanceler.create(rec.getAudioSessionId());
                if (ec != null) aec = ec.setEnabled(true) == 0;
            }
            if (NoiseSuppressor.isAvailable()) {
                ns = NoiseSuppressor.create(rec.getAudioSessionId());
                if (ns != null) ns.setEnabled(true);
            }
            rec.startRecording();
            List<Double> calib = new ArrayList<>();
            int quietMs = 0, waitedMs = 0;
            ArrayDeque<Boolean> recent = new ArrayDeque<>();
            while (running && !finishNow) {
                short[] buf = new short[FRAME];
                int n = rec.read(buf, 0, FRAME);
                if (n <= 0) continue;
                if (n < FRAME) buf = java.util.Arrays.copyOf(buf, n);
                double sum = 0;
                for (int i = 0; i < n; i++) sum += (double) buf[i] * buf[i];
                double db = 10 * Math.log10(sum / n / (32768.0 * 32768.0) + 1e-12);

                if (fired) {
                    utt.add(buf);
                    utteranceMs += 20;
                    quietMs = db > continueDb ? 0 : quietMs + 20;
                    if (quietMs >= END_SILENCE_MS || utteranceMs >= MAX_UTTERANCE_MS) break;
                    continue;
                }

                preroll.addLast(buf);
                if (preroll.size() * 20 > PREROLL_MS) preroll.removeFirst();
                if (!playing.getAsBoolean()) continue;
                if (Double.isNaN(thresholdDb)) {
                    calib.add(db);
                    if (calib.size() * 20 >= mode.calibrateMs) {
                        Collections.sort(calib);
                        echoDb = calib.get((int) (calib.size() * mode.percentile));
                        thresholdDb = Math.max(echoDb + MARGIN_DB, mode.floorDb);
                        continueDb = Math.min(thresholdDb, Math.max(echoDb + 6, CONTINUE_FLOOR_DB));
                    }
                    continue;
                }
                maxDb = Math.max(maxDb, db);
                recent.addLast(db > thresholdDb);
                if (recent.size() * 20 > TRIGGER_WINDOW_MS) recent.removeFirst();
                int above = 0;
                for (boolean b : recent) if (b) above++;
                if (above * 20 >= TRIGGER_MS) {
                    fired = true;
                    utt.addAll(preroll);   // the words that triggered it
                    utteranceMs = preroll.size() * 20;
                    preroll.clear();
                    listener.bargeIn();
                } else if (mode == Mode.LISTEN && (waitedMs += 20) >= WAIT_MS) {
                    break;
                }
            }
        } catch (Exception e) {
            error = e.toString();
        } finally {
            if (rec != null) {
                try { rec.stop(); } catch (Exception ignored) { }
                rec.release();
            }
            if (ec != null) ec.release();
            if (ns != null) ns.release();
        }
        endNs = System.nanoTime();
        if (aborted) return;
        int total = 0;
        if (fired) for (short[] s : utt) total += s.length;
        float[] pcm = new float[total];
        int k = 0;
        if (fired) for (short[] s : utt) for (short v : s) pcm[k++] = v / 32768f;
        listener.utterance(pcm, total);
    }

    /** Stops listening unless it has fired and is capturing an utterance (that ends by itself). */
    void stop() {
        if (fired) return;
        aborted = true;
        running = false;
        join();
    }

    /** Ends now: the audio so far (if the user had started talking) goes to utterance(). */
    void finish() {
        finishNow = true;
    }

    /** Stops even while capturing; utterance() is not called. */
    void abort() {
        aborted = true;
        running = false;
        join();
    }

    private void join() {
        if (Thread.currentThread() == thread) return;
        try { thread.join(1000); } catch (InterruptedException ignored) { }
    }
}
