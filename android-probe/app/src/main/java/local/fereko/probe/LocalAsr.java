package local.fereko.probe;

import java.io.File;

import local.fereko.probe.core.ReazonAsr;

/**
 * Our own Japanese speech recognizer (ReazonSpeech k2 v2, in the model pack), shared by the app.
 * Unlike Android's recognizer it takes audio we captured ourselves, so barge-in can recognize the
 * very words that interrupted ふぇれ子.
 */
final class LocalAsr {
    private LocalAsr() {}

    static final String[] FILES = {
            "encoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"};
    static final String MODE = "自前（ReazonSpeech）";
    private static final int THREADS = 4;

    private static ReazonAsr asr;

    static boolean available(File pack) {
        for (String f : FILES) if (!new File(pack, f).isFile()) return false;
        return true;
    }

    static synchronized ReazonAsr get(File pack) throws Exception {
        if (asr == null) asr = new ReazonAsr(pack, THREADS);
        return asr;
    }

    static synchronized boolean isLoaded() { return asr != null; }

    static synchronized void close() {
        if (asr != null) {
            try { asr.close(); } catch (Exception ignored) { }
        }
        asr = null;
    }
}
