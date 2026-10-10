package local.fereko.probe;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import local.fereko.probe.core.IrodoriEngine;

/**
 * DAC-VAE decoder on the Adreno GPU (LiteRT GPU delegate), fixed window of W latent frames.
 * The model (dec{W}_f32.tflite, from onnx2tf) takes [1, 32, W] channels-first and returns
 * [1, 1, W*1920]. All delegate calls run on one dedicated thread (the GPU delegate is not meant
 * to be used from several threads). Compiled GPU programs are cached on disk, so only the very
 * first creation is slow (measured 74–457 s depending on window and precision).
 */
final class GpuWindowDecoder implements IrodoriEngine.WindowDecoder {
    private final ExecutorService gpuThread = Executors.newSingleThreadExecutor(r -> new Thread(r, "gpu-decoder"));
    private final int[] windows;
    private final java.util.Map<Integer, GpuDelegate> delegates = new java.util.HashMap<>();
    private final java.util.Map<Integer, Interpreter> interpreters = new java.util.HashMap<>();
    final long createMs;

    /** @param windows window sizes to load, e.g. {44, 76}; each needs dec{W}_f32.tflite in the pack. */
    GpuWindowDecoder(File pack, File cacheDir, int[] windows, boolean fp16) throws Exception {
        this(pack, cacheDir, windows, fp16, false);
    }

    /** @param openGl run on the OpenGL ES backend instead of OpenCL (to compare how it shares the GPU with 3D) */
    GpuWindowDecoder(File pack, File cacheDir, int[] windows, boolean fp16, boolean openGl) throws Exception {
        this.windows = windows.clone();
        java.util.Arrays.sort(this.windows);
        File dir = new File(cacheDir, "gpu-cache");
        if (!dir.isDirectory() && !dir.mkdirs()) throw new IllegalStateException("mkdir " + dir);
        long t = System.nanoTime();
        for (int w : this.windows) {
            File model = new File(pack, "dec" + w + "_f32.tflite");
            if (!model.isFile()) throw new IllegalStateException(model.getName() + " がない（fereko-pack-gpu を取り込んでね）");
            String token = "dec" + w + (fp16 ? "-16" : "-32") + (openGl ? "-gl" : "") + "-" + model.length();
            gpuThread.submit(() -> {
                GpuDelegateFactory.Options go = new GpuDelegateFactory.Options()
                        .setPrecisionLossAllowed(fp16)
                        .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                        .setSerializationParams(dir.getAbsolutePath(), token);
                if (openGl) go.setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENGL);
                GpuDelegate d = new GpuDelegate(go);
                delegates.put(w, d);
                interpreters.put(w, new Interpreter(model, new Interpreter.Options().addDelegate(d)));
                return null;
            }).get();
        }
        createMs = (System.nanoTime() - t) / 1_000_000;
    }

    GpuWindowDecoder(File pack, File cacheDir, int window, boolean fp16) throws Exception {
        this(pack, cacheDir, new int[]{window}, fp16);
    }

    /** Windows whose model files are present: 44 (first chunk) and/or 76. */
    static int[] availableWindows(File pack) {
        java.util.TreeSet<Integer> v = new java.util.TreeSet<>();
        String[] names = pack.list();
        if (names != null) {
            for (String n : names) {
                java.util.regex.Matcher m = java.util.regex.Pattern.compile("dec(\\d+)_f32\\.tflite").matcher(n);
                if (m.matches()) v.add(Integer.parseInt(m.group(1)));
            }
        }
        int[] a = new int[v.size()];
        int i = 0;
        for (int w : v) a[i++] = w;
        return a;
    }

    /**
     * Windows for a decoder setting: "gpu16" / "gpu32" = the original pair (44 for the first chunk,
     * 76 after); "gpu16@36" = 36 frames for every chunk. Smaller windows are shorter GPU jobs, so
     * the 3D view stalls less while the voice is synthesized (each GPU job blocks rendering).
     */
    static int[] windowsFor(File pack, String dec) {
        int at = dec.indexOf('@');
        if (at >= 0) return new int[]{Integer.parseInt(dec.substring(at + 1))};
        java.util.List<Integer> v = new java.util.ArrayList<>();
        for (int w : availableWindows(pack)) if (w == 44 || w == 76) v.add(w);
        int[] a = new int[v.size()];
        for (int i = 0; i < a.length; i++) a[i] = v.get(i);
        return a;
    }

    @Override
    public int[] windows() { return windows.clone(); }

    @Override
    public float[] decode(float[] latentWindow, int window) throws Exception {
        Interpreter interpreter = interpreters.get(window);
        if (interpreter == null) throw new IllegalArgumentException("no GPU model for window " + window);
        Future<float[]> f = gpuThread.submit(() -> {
            // [W x 32] frame-major -> [1, 32, W] channels-first
            ByteBuffer in = ByteBuffer.allocateDirect(32 * window * 4).order(ByteOrder.nativeOrder());
            FloatBuffer inF = in.asFloatBuffer();
            for (int c = 0; c < 32; c++) for (int t = 0; t < window; t++) inF.put(latentWindow[t * 32 + c]);
            ByteBuffer out = ByteBuffer.allocateDirect(window * IrodoriEngine.HOP * 4).order(ByteOrder.nativeOrder());
            interpreter.run(in, out);
            out.rewind();
            float[] a = new float[window * IrodoriEngine.HOP];
            out.asFloatBuffer().get(a);
            return a;
        });
        return f.get();
    }

    @Override
    public void close() {
        try {
            gpuThread.submit(() -> {
                for (Interpreter i : interpreters.values()) i.close();
                for (GpuDelegate d : delegates.values()) d.close();
                return null;
            }).get();
        } catch (Exception ignored) {
        } finally {
            gpuThread.shutdown();
        }
    }
}
