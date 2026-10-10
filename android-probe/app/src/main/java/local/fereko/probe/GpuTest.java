package local.fereko.probe;

import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import local.fereko.probe.core.IrodoriEngine;

/**
 * DAC-VAE decoder on the Adreno GPU through LiteRT (TFLite) GPU delegate.
 * Models (fereko-pack-gpu): dec{W}_f32.tflite, converted on PC with onnx2tf from the decoder with
 * the Snake reciprocals constant-folded; input [1, 32, W] (channels first), output [1, 1, W*1920].
 * fp16 = GPU computes in half precision (precision loss allowed); fp32 = full precision on the GPU.
 * Reference: the same model on the CPU (LiteRT, 6 threads).
 */
final class GpuTest {
    private GpuTest() {}

    static JSONObject run(File pack, int window, boolean fp16, NpuTest.Log log, NpuTest.Step step) throws Exception {
        JSONObject o = new JSONObject();
        o.put("backend", "gpu" + window + (fp16 ? "-16bit" : "-32bit"));
        o.put("window", window);
        File model = new File(pack, "dec" + window + "_f32.tflite");
        if (!model.isFile()) throw new IllegalStateException(model.getName() + " がない（fereko-pack-gpu を取り込んでね）");

        // latent frames, transposed to [1, 32, W]
        float[] lat = IrodoriEngine.readF32(new File(pack, "fixtures/parity_latent.f32"));
        int frames = Math.min(window, lat.length / 32);
        ByteBuffer in = ByteBuffer.allocateDirect(32 * window * 4).order(ByteOrder.nativeOrder());
        FloatBuffer inF = in.asFloatBuffer();
        for (int c = 0; c < 32; c++) for (int t = 0; t < window; t++) inF.put(t < frames ? lat[t * 32 + c] : 0f);
        int outLen = window * IrodoriEngine.HOP;

        GpuDelegateFactory.Options go = new GpuDelegateFactory.Options()
                .setPrecisionLossAllowed(fp16)
                .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
        step.step("gpu_create", o.optString("backend"));
        long t = System.nanoTime();
        float[] out;
        try (GpuDelegate gpu = new GpuDelegate(go);
             Interpreter it = new Interpreter(model, new Interpreter.Options().addDelegate(gpu))) {
            o.put("create_ms", (System.nanoTime() - t) / 1_000_000);
            o.put("input_shape", java.util.Arrays.toString(it.getInputTensor(0).shape()));
            step.step("first_run", "created in " + o.optLong("create_ms") + " ms");
            out = invoke(it, in, outLen);
            t = System.nanoTime();
            out = invoke(it, in, outLen);
            o.put("first_ms", (System.nanoTime() - t) / 1_000_000);
            step.step("timed_runs", null);
            t = System.nanoTime();
            for (int i = 0; i < 5; i++) out = invoke(it, in, outLen);
            o.put("ms", (System.nanoTime() - t) / 5_000_000);
        }
        step.step("cpu_reference", model.getName());
        float[] ref;
        try (Interpreter cpu = new Interpreter(model, new Interpreter.Options().setNumThreads(6))) {
            t = System.nanoTime();
            ref = invoke(cpu, in, outLen);
            o.put("cpu_ms", (System.nanoTime() - t) / 1_000_000);
        }
        double sig = 0, err = 0;
        for (int i = 0; i < outLen; i++) {
            double d = ref[i] - out[i];
            sig += ref[i] * (double) ref[i];
            err += d * d;
        }
        o.put("snr_db", err == 0 ? 999 : 10 * Math.log10(sig / err));
        double audioMs = (window - 12) * 40.0;   // new frames per call with 6 frames of context each side
        o.put("rtf_new_frames", o.optLong("ms") / audioMs);
        log.line(String.format(java.util.Locale.US, "GPU %s: 作成 %dms, 平均 %dms（CPU %dms）, SNR %.1fdB",
                o.optString("backend"), o.optLong("create_ms"), o.optLong("ms"), o.optLong("cpu_ms"), o.optDouble("snr_db")));
        return o;
    }

    private static float[] invoke(Interpreter it, ByteBuffer in, int outLen) {
        in.rewind();
        ByteBuffer out = ByteBuffer.allocateDirect(outLen * 4).order(ByteOrder.nativeOrder());
        it.run(in, out);
        out.rewind();
        float[] a = new float[outLen];
        out.asFloatBuffer().get(a);
        return a;
    }
}
