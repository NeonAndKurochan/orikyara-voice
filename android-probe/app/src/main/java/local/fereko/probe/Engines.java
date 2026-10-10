package local.fereko.probe;

import java.io.File;

import local.fereko.probe.core.IrodoriEngine;

/**
 * One engine for the whole app: the FP32 graphs take ~3-4 GB, so the probe screen and the
 * chat screen must share it rather than each loading their own.
 */
final class Engines {
    private Engines() {}

    private static IrodoriEngine engine;
    private static String key;
    /** Where the GPU delegate caches its compiled programs (set by the activities). */
    static File cacheDir;

    static synchronized IrodoriEngine get(File pack, int threads, IrodoriEngine.Accel accel, String dit) throws Exception {
        return get(pack, threads, accel, dit, "");
    }

    static synchronized IrodoriEngine get(File pack, int threads, IrodoriEngine.Accel accel, String dit, String dec) throws Exception {
        String k = threads + "/" + accel + "/" + dit + "/" + dec;
        if (engine != null && k.equals(key)) return engine;
        close();
        if (dec.startsWith("gpu") || dec.startsWith("gl")) {
            // ONNX for everything but the decoder, which runs on the GPU in 76-frame windows
            engine = new IrodoriEngine(pack, threads, accel, dit, "");
            try {
                // "gl16..." = the same models on the OpenGL ES backend (test setting)
                engine.setWindowDecoder(new GpuWindowDecoder(pack, cacheDir, GpuWindowDecoder.windowsFor(pack, dec),
                        dec.startsWith("gpu16") || dec.startsWith("gl16"), dec.startsWith("gl")));
            } catch (Exception e) {
                close();
                throw e;
            }
        } else {
            engine = new IrodoriEngine(pack, threads, accel, dit, dec);
        }
        key = k;
        return engine;
    }

    static synchronized boolean isLoaded(int threads, IrodoriEngine.Accel accel, String dit) {
        return isLoaded(threads, accel, dit, "");
    }

    static synchronized boolean isLoaded(int threads, IrodoriEngine.Accel accel, String dit, String dec) {
        return engine != null && (threads + "/" + accel + "/" + dit + "/" + dec).equals(key);
    }

    /** Decoder variants present in the pack: "" (FP32) plus any dacvae_decode.<variant>.onnx. */
    static String[] decVariants(File pack) {
        java.util.List<String> v = new java.util.ArrayList<>();
        v.add("");
        if (new File(pack, "dec76_f32.tflite").isFile()) {
            v.add("gpu16");
            v.add("gpu32");
        }
        for (int w : GpuWindowDecoder.availableWindows(pack)) if (w != 44 && w != 76) v.add("gpu16@" + w);
        return v.toArray(new String[0]);
    }

    static synchronized void close() {
        if (engine != null && engine.windowDecoder() != null) engine.windowDecoder().close();
        if (engine != null) engine.close();
        engine = null;
        key = null;
    }

    /** DiT variants present in the pack: "" (FP32) plus any dit_step.<variant>.onnx. */
    static String[] ditVariants(File pack) {
        java.util.List<String> v = new java.util.ArrayList<>();
        v.add("");
        for (String name : new String[]{"int8dyn", "nbits8", "nbits4"}) {
            if (new File(pack, "dit_step." + name + ".onnx").isFile()) v.add(name);
        }
        return v.toArray(new String[0]);
    }
}
