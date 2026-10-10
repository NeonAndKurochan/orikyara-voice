package local.fereko.probe;

import org.json.JSONObject;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtEpDevice;
import ai.onnxruntime.OrtHardwareDevice;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import local.fereko.probe.core.IrodoriEngine;

/**
 * Feasibility test: run the static-shape decoder (dacvae_decode.static{W}.onnx, from fereko-pack-npu)
 * on Qualcomm's QNN execution provider plugin (com.qualcomm.qti:onnxruntime-android-qnn) — HTP = NPU,
 * or the Adreno GPU backend — and compare speed and output with the CPU provider on the same input.
 *
 * On PC the same plugin (onnxruntime-qnn 2.6.0, QNN 2.50) compiled the whole decoder for HTP v79
 * with CPU fallback disabled (tools/qnn_compile_check.py), so "strict" is tried first.
 */
final class NpuTest {
    private NpuTest() {}

    interface Log { void line(String s); }

    /** Called before each step, so a hang in native code leaves the last step on record. */
    interface Step { void step(String name, String detail); }

    private static final String REG_NAME = "qnn_plugin";
    private static boolean registered;

    /** Registers the plugin library once and logs every EP device it exposes. */
    static synchronized List<OrtEpDevice> qnnDevices(String nativeLibDir, Log log, Step step) throws Exception {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        if (!registered) {
            // The HTP backend loads its DSP-side skeleton library through FastRPC from these paths.
            android.system.Os.setenv("ADSP_LIBRARY_PATH",
                    nativeLibDir + ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp", true);
            step.step("register_plugin", null);
            env.registerExecutionProviderLibrary(REG_NAME, nativeLibDir + "/libonnxruntime_providers_qnn.so");
            registered = true;
            step.step("list_devices", null);
            StringBuilder all = new StringBuilder();
            for (OrtEpDevice d : env.getEpDevices()) {
                String s = d.getEpName() + "/" + d.getDevice().getType() + "/" + d.getDevice().getVendor()
                        + " dev" + d.getDevice().getMetadata() + " ep" + d.getEpMetadata();
                log.line("EP device: " + s + " " + d.getDevice().getMetadata());
                all.append(s).append("; ");
            }
            step.step("devices", all.toString());
        }
        List<OrtEpDevice> out = new ArrayList<>();
        for (OrtEpDevice d : env.getEpDevices()) if (d.getEpName().toUpperCase().contains("QNN")) out.add(d);
        return out;
    }

    static JSONObject run(File pack, String backend, int window, int cpuThreads, String nativeLibDir, Log log) throws Exception {
        return run(pack, backend, window, cpuThreads, nativeLibDir, log, (n, d) -> { });
    }

    static JSONObject run(File pack, String backend, int window, int cpuThreads, String nativeLibDir, Log log, Step step) throws Exception {
        JSONObject o = new JSONObject();
        o.put("backend", backend);
        o.put("window", window);
        File model;
        if (backend.startsWith("htp-ctx:")) {
            model = new File(pack, backend.substring("htp-ctx:".length()));
        } else {
            String suffix = backend.equals("htp-c2d") ? "c2d" : backend.equals("htp-ctx") ? "c2d.qnnctx" : "";
            model = new File(pack, "dacvae_decode.static" + window + suffix + ".onnx");
        }
        File cpuModel = new File(pack, "dacvae_decode.static" + window + (backend.equals("htp") ? "" : "c2d") + ".onnx");
        if (!model.isFile() || !cpuModel.isFile()) {
            throw new IllegalStateException(model.getName() + " がない（fereko-pack-npu / fereko-pack-npu2 を取り込んでね）");
        }

        // Real latent frames (the parity fixture), so the output is meaningful audio.
        float[] lat = IrodoriEngine.readF32(new File(pack, "fixtures/parity_latent.f32"));
        float[] in = new float[window * IrodoriEngine.LATENT_DIM];
        System.arraycopy(lat, 0, in, 0, Math.min(in.length, lat.length));

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        float[] ref;
        long cpuMs;
        step.step("cpu_reference", cpuModel.getName());
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
            so.setIntraOpNumThreads(cpuThreads);
            so.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
            try (OrtSession cpu = env.createSession(cpuModel.getAbsolutePath(), so)) {
                ref = decode(env, cpu, in, window);
                long t = System.nanoTime();
                for (int i = 0; i < 3; i++) decode(env, cpu, in, window);
                cpuMs = (System.nanoTime() - t) / 3_000_000;
            }
        }
        o.put("cpu_ms", cpuMs);

        o.put("model", model.getName());
        OrtHardwareDevice.OrtHardwareDeviceType want = backend.equals("gpu")
                ? OrtHardwareDevice.OrtHardwareDeviceType.GPU : OrtHardwareDevice.OrtHardwareDeviceType.NPU;
        OrtEpDevice dev = null;
        for (OrtEpDevice d : qnnDevices(nativeLibDir, log, step)) if (d.getDevice().getType() == want) dev = d;
        if (dev == null) throw new IllegalStateException("QNN の " + want + " デバイスが見つからない");
        o.put("device", dev.getDevice().getType() + " " + dev.getDevice().getVendor());

        Map<String, String> qo = new HashMap<>();
        qo.put("backend_path", nativeLibDir + (backend.equals("gpu") ? "/libQnnGpu.so" : "/libQnnHtp.so"));
        if (!backend.equals("gpu")) {
            qo.put("htp_performance_mode", "burst");
            qo.put("enable_htp_fp16_precision", "1");
            qo.put("htp_graph_finalization_optimization_mode", "3");
        }
        // strict = every node on QNN (no CPU fallback); if that fails, measure the partial split too.
        for (boolean strict : new boolean[]{true, false}) {
            try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
                so.setIntraOpNumThreads(cpuThreads);   // for any ops QNN leaves to the CPU
                so.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
                if (strict) so.addConfigEntry("session.disable_cpu_ep_fallback", "1");
                so.addExecutionProvider(Collections.singletonList(dev), qo);
                long t = System.nanoTime();
                step.step("create_session", backend + (strict ? " strict" : " partial") + " " + model.getName());
                try (OrtSession q = env.createSession(model.getAbsolutePath(), so)) {
                    o.put("all_on_qnn", strict);
                    o.put("create_ms", (System.nanoTime() - t) / 1_000_000);
                    step.step("first_run", "created in " + o.optLong("create_ms") + " ms");
                    t = System.nanoTime();
                    float[] out = decode(env, q, in, window);
                    o.put("first_ms", (System.nanoTime() - t) / 1_000_000);
                    step.step("timed_runs", "first run " + o.optLong("first_ms") + " ms");
                    t = System.nanoTime();
                    int n = 5;
                    for (int i = 0; i < n; i++) decode(env, q, in, window);
                    long ms = (System.nanoTime() - t) / (n * 1_000_000L);
                    o.put("ms", ms);
                    double sig = 0, err = 0, maxd = 0;
                    for (int i = 0; i < ref.length; i++) {
                        double d = ref[i] - out[i];
                        sig += ref[i] * (double) ref[i];
                        err += d * d;
                        maxd = Math.max(maxd, Math.abs(d));
                    }
                    o.put("snr_db", err == 0 ? 999 : 10 * Math.log10(sig / err));
                    o.put("maxdiff", maxd);
                    // Each call of a W-frame window yields W - 12 new frames (6 frames of context each side).
                    double audioMs = (window - 12) * 40.0;
                    o.put("rtf_new_frames", ms / audioMs);
                    o.put("cpu_rtf_new_frames", cpuMs / audioMs);
                    break;
                } catch (Exception e) {
                    if (!strict) throw e;
                    o.put("strict_error", String.valueOf(e.getMessage()));
                    log.line("全部を " + backend + " には載せられなかった: " + e.getMessage());
                }
            }
        }
        log.line(String.format(java.util.Locale.US, "NPU試験 %s W%d: %s 作成 %dms, 1回 %dms（CPU %dms）, SNR %.1fdB",
                backend, window, o.optBoolean("all_on_qnn") ? "全部QNN" : "一部CPU",
                o.optLong("create_ms"), o.optLong("ms"), cpuMs, o.optDouble("snr_db")));
        return o;
    }

    /**
     * Smallest possible HTP check: one 1x1 Conv + Relu (tiny.onnx, input x [1,32,1,44]).
     * ctx = context compiled on PC for v73, online = compiled on the phone. If even this hangs,
     * the problem is the DSP setup (FastRPC / skeleton libraries), not the decoder.
     */
    static JSONObject tiny(File pack, boolean ctx, String nativeLibDir, Log log, Step step) throws Exception {
        JSONObject o = new JSONObject();
        o.put("backend", ctx ? "tiny-ctx" : "tiny-online");
        o.put("window", 44);
        File cpuModel = new File(pack, "tiny.onnx");
        File model = ctx ? new File(pack, "tiny_v73.v73.qnnctx.onnx") : cpuModel;
        if (!model.isFile()) throw new IllegalStateException(model.getName() + " がない（fereko-pack-npu4 を取り込んでね）");
        float[] in = new float[32 * 44];
        for (int i = 0; i < in.length; i++) in[i] = (float) Math.sin(i * 0.1);
        long[] shape = {1, 32, 1, 44};
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        float[] ref;
        step.step("cpu_reference", cpuModel.getName());
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions();
             OrtSession cpu = env.createSession(cpuModel.getAbsolutePath(), so)) {
            ref = runX(env, cpu, in, shape);
        }
        OrtEpDevice dev = null;
        for (OrtEpDevice d : qnnDevices(nativeLibDir, log, step)) {
            if (d.getDevice().getType() == OrtHardwareDevice.OrtHardwareDeviceType.NPU) dev = d;
        }
        if (dev == null) throw new IllegalStateException("QNN の NPU デバイスが見つからない");
        Map<String, String> qo = new HashMap<>();
        qo.put("backend_path", nativeLibDir + "/libQnnHtp.so");
        qo.put("enable_htp_fp16_precision", "1");
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
            so.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
            so.addConfigEntry("session.disable_cpu_ep_fallback", "1");
            so.addExecutionProvider(Collections.singletonList(dev), qo);
            step.step("create_session", o.optString("backend") + " " + model.getName());
            long t = System.nanoTime();
            try (OrtSession q = env.createSession(model.getAbsolutePath(), so)) {
                o.put("create_ms", (System.nanoTime() - t) / 1_000_000);
                step.step("first_run", "created in " + o.optLong("create_ms") + " ms");
                t = System.nanoTime();
                float[] out = runX(env, q, in, shape);
                o.put("first_ms", (System.nanoTime() - t) / 1_000_000);
                double maxd = 0;
                for (int i = 0; i < ref.length; i++) maxd = Math.max(maxd, Math.abs(ref[i] - out[i]));
                o.put("maxdiff", maxd);
                o.put("all_on_qnn", true);
            }
        }
        log.line("小テスト " + o.optString("backend") + ": 作成 " + o.optLong("create_ms") + "ms, 1回目 " + o.optLong("first_ms")
                + "ms, 差 " + o.optDouble("maxdiff"));
        return o;
    }

    /**
     * Any EP-context model: input shape read from the model, fed with real latent frames
     * (fixtures/parity_latent.f32, transposed/padded as needed). Compared with the CPU model of the
     * same name without ".v73.qnnctx" when that file is present; otherwise only speed is measured.
     */
    static JSONObject ctxGeneric(File pack, String ctxName, String nativeLibDir, Log log, Step step) throws Exception {
        JSONObject o = new JSONObject();
        o.put("backend", "ctx:" + ctxName.replace(".qnnctx.onnx", ""));
        o.put("window", 0);
        File model = new File(pack, ctxName);
        File cpuModel = new File(pack, ctxName.replaceAll("\\.v7\\d\\.qnnctx\\.onnx$", ".onnx").replace(".qnnctx.onnx", ".onnx"));
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtEpDevice dev = null;
        for (OrtEpDevice d : qnnDevices(nativeLibDir, log, step)) {
            if (d.getDevice().getType() == OrtHardwareDevice.OrtHardwareDeviceType.NPU) dev = d;
        }
        if (dev == null) throw new IllegalStateException("QNN の NPU デバイスが見つからない");
        Map<String, String> qo = new HashMap<>();
        qo.put("backend_path", nativeLibDir + "/libQnnHtp.so");
        qo.put("htp_performance_mode", "burst");
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
            so.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
            so.addConfigEntry("session.disable_cpu_ep_fallback", "1");
            so.addExecutionProvider(Collections.singletonList(dev), qo);
            step.step("create_session", ctxName);
            long t = System.nanoTime();
            try (OrtSession q = env.createSession(model.getAbsolutePath(), so)) {
                o.put("create_ms", (System.nanoTime() - t) / 1_000_000);
                String inName = q.getInputNames().iterator().next();
                long[] shape = ((ai.onnxruntime.TensorInfo) q.getInputInfo().get(inName).getInfo()).getShape();
                o.put("input_shape", java.util.Arrays.toString(shape));
                int n = 1;
                for (long s : shape) n *= (int) s;
                float[] lat = IrodoriEngine.readF32(new File(pack, "fixtures/parity_latent.f32"));
                float[] in = new float[n];
                System.arraycopy(lat, 0, in, 0, Math.min(n, lat.length));   // [1, W, 32] layout: first W frames
                step.step("first_run", "created in " + o.optLong("create_ms") + " ms, input " + o.optString("input_shape"));
                t = System.nanoTime();
                float[] out = runNamed(env, q, inName, in, shape);
                o.put("first_ms", (System.nanoTime() - t) / 1_000_000);
                step.step("timed_runs", "first run " + o.optLong("first_ms") + " ms");
                t = System.nanoTime();
                for (int i = 0; i < 5; i++) runNamed(env, q, inName, in, shape);
                o.put("ms", (System.nanoTime() - t) / 5_000_000);
                o.put("all_on_qnn", true);
                if (cpuModel.isFile() && !cpuModel.equals(model)) {
                    step.step("cpu_reference", cpuModel.getName());
                    try (OrtSession.SessionOptions cs = new OrtSession.SessionOptions()) {
                        cs.setIntraOpNumThreads(6);
                        try (OrtSession cpu = env.createSession(cpuModel.getAbsolutePath(), cs)) {
                            t = System.nanoTime();
                            float[] ref = runNamed(env, cpu, inName, in, shape);
                            o.put("cpu_ms", (System.nanoTime() - t) / 1_000_000);
                            double sig = 0, err = 0;
                            for (int i = 0; i < ref.length && i < out.length; i++) {
                                double d = ref[i] - out[i];
                                sig += ref[i] * (double) ref[i];
                                err += d * d;
                            }
                            o.put("snr_db", err == 0 ? 999 : 10 * Math.log10(sig / err));
                        }
                    }
                }
            }
        }
        log.line("ctx " + ctxName + ": 作成 " + o.optLong("create_ms") + "ms, 1回目 " + o.optLong("first_ms") + "ms, 平均 " + o.optLong("ms") + "ms");
        return o;
    }

    private static float[] runNamed(OrtEnvironment env, OrtSession s, String inName, float[] in, long[] shape) throws Exception {
        FloatBuffer b = ByteBuffer.allocateDirect(in.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        b.put(in).rewind();
        try (OnnxTensor t = OnnxTensor.createTensor(env, b, shape);
             OrtSession.Result r = s.run(Collections.singletonMap(inName, t))) {
            FloatBuffer w = ((OnnxTensor) r.get(0)).getFloatBuffer();
            float[] out = new float[w.remaining()];
            w.get(out);
            return out;
        }
    }

    private static float[] runX(OrtEnvironment env, OrtSession s, float[] in, long[] shape) throws Exception {
        FloatBuffer b = ByteBuffer.allocateDirect(in.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        b.put(in).rewind();
        try (OnnxTensor t = OnnxTensor.createTensor(env, b, shape);
             OrtSession.Result r = s.run(Collections.singletonMap("x", t))) {
            FloatBuffer w = ((OnnxTensor) r.get(0)).getFloatBuffer();
            float[] out = new float[w.remaining()];
            w.get(out);
            return out;
        }
    }

    private static float[] decode(OrtEnvironment env, OrtSession s, float[] in, int window) throws Exception {
        FloatBuffer b = ByteBuffer.allocateDirect(in.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        b.put(in).rewind();
        try (OnnxTensor t = OnnxTensor.createTensor(env, b, new long[]{1, window, IrodoriEngine.LATENT_DIM});
             OrtSession.Result r = s.run(Collections.singletonMap("latent", t))) {
            FloatBuffer w = ((OnnxTensor) r.get(0)).getFloatBuffer();
            float[] out = new float[w.remaining()];
            w.get(out);
            return out;
        }
    }
}
