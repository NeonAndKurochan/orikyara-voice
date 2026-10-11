package local.fereko.probe;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import local.fereko.probe.core.MiniJson;

/**
 * One character the user can talk to. ふぇれ子 is built in (persona and voice caption in the APK's
 * assets, her data where it always was: files/chat/, her registered voice and VRM in the model pack).
 * Others are folders files/chars/&lt;id&gt;/ with:
 *   character.json  {"name": "…", "caption": "voice description", "avatar": "vrm" | "image" | "live2d"}
 *   persona.md      the system prompt (same structure as chat/persona.md, incl. the 好感度 rules)
 *   speaker.f32 + speaker.json   registered voice (optional; without it the caption alone sets the voice)
 *   *.vrm, *.vrma   3D model and motions (optional)
 *   flat/rig.json + layers   the picture as 2D parts, moved by the same motions (optional, needs the VRM;
 *                   made with tools/flat_bake.py)
 *   stand.png|webp  still picture (optional; also the fallback while Live2D isn't supported)
 * Each character has its own history, memo, archive, affection and greeting time; app-wide
 * settings (Claude model, recognizer, decoder, tempo, background…) are shared.
 */
final class CharacterProfile {
    static final String BUILTIN = "fereko";

    final String id, name, persona, defaultCaption, avatarType;
    /** null for ふぇれ子 (built in) */
    final File dir;
    private final File filesDir;

    private CharacterProfile(String id, String name, String persona, String caption, String avatarType, File dir, File filesDir) {
        this.id = id;
        this.name = name;
        this.persona = persona;
        this.defaultCaption = caption;
        this.avatarType = avatarType;
        this.dir = dir;
        this.filesDir = filesDir;
    }

    boolean builtin() { return dir == null; }

    static File charsDir(Context c) { return new File(c.getFilesDir(), "chars"); }

    /** ふぇれ子 first, then the character folders that have a character.json and persona.md. */
    static List<CharacterProfile> all(Context c) {
        List<CharacterProfile> out = new ArrayList<>();
        out.add(builtin(c));
        File[] ds = charsDir(c).listFiles(File::isDirectory);
        if (ds != null) {
            java.util.Arrays.sort(ds);
            for (File d : ds) {
                try {
                    out.add(fromDir(c, d));
                } catch (Exception ignored) {
                    // incomplete folder: not listed
                }
            }
        }
        return out;
    }

    static CharacterProfile load(Context c, String id) {
        if (!BUILTIN.equals(id)) {
            File d = new File(charsDir(c), id);
            try {
                return fromDir(c, d);
            } catch (Exception ignored) {
                // gone or broken: back to ふぇれ子
            }
        }
        return builtin(c);
    }

    private static CharacterProfile builtin(Context c) {
        return new CharacterProfile(BUILTIN, "ふぇれ子", asset(c, "persona.md"), asset(c, "voice.txt").trim(), "vrm", null, c.getFilesDir());
    }

    private static CharacterProfile fromDir(Context c, File d) throws IOException {
        Map<String, Object> j = MiniJson.obj(MiniJson.parse(read(new File(d, "character.json"))));
        String persona = read(new File(d, "persona.md"));
        Object name = j.get("name"), cap = j.get("caption"), av = j.get("avatar");
        return new CharacterProfile(d.getName(), name == null ? d.getName() : (String) name, persona,
                cap == null ? "" : ((String) cap).trim(), av == null ? "image" : (String) av, d, c.getFilesDir());
    }

    // ---- per-character data ----

    private File dataDir() { return builtin() ? new File(filesDir, "chat") : dir; }

    File historyFile() { return new File(dataDir(), "history.json"); }

    File memoFile() { return new File(dataDir(), "memory.md"); }

    File archiveFile() { return new File(dataDir(), "history-archive.jsonl"); }

    /** Preference key for per-character values (ふぇれ子 keeps the original keys). */
    String key(String k) { return builtin() ? k : id + "." + k; }

    /** Folder with speaker.f32 (registered voice): the pack for ふぇれ子. */
    File voiceDir(File pack) { return builtin() ? pack : dir; }

    /** The 2D rig (picture layers moved by the same motions), or null: modelDir/flat/rig.json. */
    File flatRig(File pack) { return AvatarView.findFlatRig(modelDir(pack)); }

    /** Folder the 3D page reads the model and motions from. */
    File modelDir(File pack) { return builtin() ? pack : dir; }

    /** The VRM to show, or null (no 3D, or an avatar type that isn't 3D). */
    File vrm(File pack) {
        if (!"vrm".equals(avatarType)) return null;
        if (builtin()) return AvatarView.findVrm(pack);
        File[] fs = dir.listFiles((x, n) -> n.endsWith(".vrm"));
        if (fs == null || fs.length == 0) return null;
        File best = fs[0];
        for (File f : fs) if (f.lastModified() > best.lastModified()) best = f;
        return best;
    }

    /** Still picture (cut out, transparent background), or null. */
    Bitmap stand(Context c) {
        if (builtin()) {
            try (InputStream in = c.getAssets().open("fereko_stand_cut.webp")) {
                return BitmapFactory.decodeStream(in);
            } catch (IOException e) {
                return null;
            }
        }
        for (String n : new String[]{"stand.png", "stand.webp"}) {
            File f = new File(dir, n);
            if (f.isFile()) return BitmapFactory.decodeFile(f.getAbsolutePath());
        }
        return null;
    }

    private static String asset(Context c, String name) {
        try (InputStream in = c.getAssets().open(name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            return "";
        }
    }

    private static String read(File f) throws IOException {
        return new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);
    }
}
