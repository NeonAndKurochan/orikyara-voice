package local.fereko.probe;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;

import java.util.ArrayList;

/**
 * Japanese speech-to-text with Android's SpeechRecognizer. Prefers the on-device recognizer
 * (audio stays on the phone); falls back to the default recognizer service (which may use the
 * network) only when on-device recognition is unavailable, and says so through state().
 * Must be used from the main thread.
 */
final class VoiceInput {
    interface Callback {
        /** Raw recognizer events for diagnostics, e.g. "ready", "begin", "end", "error 7". */
        default void event(String e) { }
        void state(String s);
        void partial(String s);
        /** Final text; sttMs = end of speech to final result. */
        void result(String text, long sttMs, String mode);
        void error(String s);
    }

    private final Activity activity;
    private final Callback cb;
    private SpeechRecognizer recognizer;
    private Intent intent;
    String mode = "";
    boolean listening;
    private long endOfSpeechNs;

    VoiceInput(Activity activity, Callback cb) {
        this.activity = activity;
        this.cb = cb;
    }

    private void create() {
        if (recognizer != null) return;
        if (SpeechRecognizer.isOnDeviceRecognitionAvailable(activity)) {
            recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(activity);
            mode = "端末内";
        } else if (SpeechRecognizer.isRecognitionAvailable(activity)) {
            recognizer = SpeechRecognizer.createSpeechRecognizer(activity);
            mode = "標準（ネット経由のことあり）";
            cb.state("端末内の音声認識が使えないから、標準の音声認識を使うよ（Google のサーバーに音声が送られることがある）");
        } else {
            cb.error("この端末では音声認識が使えないみたい");
            return;
        }
        intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ja-JP")
                .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                .putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
                .putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);
        recognizer.setRecognitionListener(new RecognitionListener() {
            @Override public void onReadyForSpeech(Bundle params) {
                cb.event("ready");
                cb.state("聞いてるよ…（" + mode + "）");
            }
            @Override public void onBeginningOfSpeech() { cb.event("begin"); }
            @Override public void onRmsChanged(float rmsdB) { }
            @Override public void onBufferReceived(byte[] buffer) { }
            @Override public void onEndOfSpeech() {
                cb.event("end");
                endOfSpeechNs = System.nanoTime();
                cb.state("認識中…");
            }
            @Override public void onError(int error) {
                cb.event("error " + error);
                listening = false;
                if (error == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE && Build.VERSION.SDK_INT >= 33) {
                    recognizer.triggerModelDownload(intent);
                    cb.error("日本語の端末内モデルをダウンロードし始めたよ。終わったらもう一度押してね");
                } else if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                    cb.error("聞き取れなかった");
                } else {
                    cb.error("音声認識のエラー " + error);
                }
            }
            @Override public void onResults(Bundle results) {
                listening = false;
                ArrayList<String> r = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                String text = r == null || r.isEmpty() ? "" : r.get(0).trim();
                cb.event("results " + text.length() + " chars");
                long ms = endOfSpeechNs == 0 ? -1 : (System.nanoTime() - endOfSpeechNs) / 1_000_000;
                if (text.isEmpty()) cb.error("聞き取れなかった");
                else cb.result(text, ms, mode);
            }
            @Override public void onPartialResults(Bundle partial) {
                ArrayList<String> r = partial.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (r != null && !r.isEmpty()) cb.partial(r.get(0));
            }
            @Override public void onEvent(int eventType, Bundle params) { }
        });
    }

    void start() {
        create();
        if (recognizer == null) return;
        endOfSpeechNs = 0;
        listening = true;
        recognizer.startListening(intent);
    }

    /**
     * Recognize audio we capture ourselves (16 kHz mono PCM16 from the barge-in detector) instead of
     * the recognizer's own microphone. Android 13+ (EXTRA_AUDIO_SOURCE). The recognizer reads until
     * the write side of the pipe is closed.
     * @return false if not supported, so the caller can fall back to start()
     */
    boolean startWithAudio(android.os.ParcelFileDescriptor audio, int sampleRate) {
        if (Build.VERSION.SDK_INT < 33) return false;
        create();
        if (recognizer == null) return false;
        Intent i = new Intent(intent)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, audio)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING, android.media.AudioFormat.ENCODING_PCM_16BIT)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, sampleRate)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1);
        endOfSpeechNs = 0;
        listening = true;
        recognizer.startListening(i);
        return true;
    }

    void stop() {
        if (recognizer != null && listening) recognizer.stopListening();
    }

    void destroy() {
        if (recognizer != null) recognizer.destroy();
        recognizer = null;
        listening = false;
    }
}
