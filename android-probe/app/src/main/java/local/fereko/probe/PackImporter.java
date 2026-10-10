package local.fereko.probe;

import android.content.ContentResolver;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import local.fereko.probe.core.MiniJson;

/**
 * Copies the pack folder chosen with the system folder picker into app-private storage.
 * Only files listed in pack.json are copied; each is size- and SHA-256-checked while streaming.
 * Paths with "..", absolute paths, or unknown entries are rejected.
 */
final class PackImporter {
    interface Progress { void update(String file, long done, long total); }

    static final class Report {
        int files;
        long bytes;
        long millis;
    }

    private final ContentResolver cr;
    private final Uri tree;

    PackImporter(ContentResolver cr, Uri tree) {
        this.cr = cr;
        this.tree = tree;
    }

    Report importTo(File dest, Progress progress) throws Exception {
        long t0 = System.currentTimeMillis();
        String rootId = DocumentsContract.getTreeDocumentId(tree);
        Map<String, String> top = list(rootId);
        String packJsonId = top.get("pack.json");
        if (packJsonId == null) throw new IOException("選んだフォルダに pack.json がありません");
        String packJson = readText(packJsonId);
        Map<String, Object> packObj = MiniJson.obj(MiniJson.parse(packJson));
        List<Object> files = MiniJson.arr(packObj.get("files"));
        // An overlay pack (e.g. quantized DiT files) adds files to the installed pack instead of replacing it.
        boolean overlay = Boolean.TRUE.equals(packObj.get("overlay"));
        if (overlay && !new File(dest, "pack.json").isFile()) throw new IOException("先に本体のモデル（fereko-pack）を取り込んでね");

        long total = 0;
        for (Object o : files) total += ((Double) MiniJson.obj(o).get("size")).longValue();

        File tmp = new File(dest.getParentFile(), dest.getName() + ".importing");
        deleteRecursive(tmp);
        if (!tmp.mkdirs()) throw new IOException("cannot create " + tmp);

        long done = 0;
        byte[] buf = new byte[1 << 20];
        Map<String, Map<String, String>> dirCache = new HashMap<>();
        dirCache.put("", top);
        for (Object o : files) {
            Map<String, Object> f = MiniJson.obj(o);
            String path = (String) f.get("path");
            long size = ((Double) f.get("size")).longValue();
            String sha = (String) f.get("sha256");
            if (path.startsWith("/") || path.contains("..") || path.contains("\\")) throw new IOException("bad path " + path);

            int slash = path.lastIndexOf('/');
            String dir = slash < 0 ? "" : path.substring(0, slash);
            String name = path.substring(slash + 1);
            Map<String, String> entries = dirCache.get(dir);
            if (entries == null) {
                String dirId = top.get(dir);
                if (dirId == null) throw new IOException("フォルダがありません: " + dir);
                entries = list(dirId);
                dirCache.put(dir, entries);
            }
            String docId = entries.get(name);
            if (docId == null) throw new IOException("ファイルがありません: " + path);

            File out = new File(tmp, path);
            File parent = out.getParentFile();
            if (parent != null && !parent.isDirectory() && !parent.mkdirs()) throw new IOException("mkdir " + parent);
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            long n = 0;
            long lastReport = 0;
            try (InputStream in = cr.openInputStream(DocumentsContract.buildDocumentUriUsingTree(tree, docId));
                 OutputStream os = new FileOutputStream(out)) {
                if (in == null) throw new IOException("cannot open " + path);
                int r;
                while ((r = in.read(buf)) > 0) {
                    os.write(buf, 0, r);
                    md.update(buf, 0, r);
                    n += r;
                    if (n > size) throw new IOException("サイズが違います: " + path);
                    if (n - lastReport > (64 << 20)) {
                        lastReport = n;
                        progress.update(path, done + n, total);
                    }
                }
            }
            if (n != size) throw new IOException("サイズが違います: " + path + " " + n + " != " + size);
            String got = hex(md.digest());
            if (!got.equals(sha)) throw new IOException("ハッシュが違います: " + path);
            done += n;
            progress.update(path, done, total);
        }
        if (overlay) {
            for (Object o : files) {
                String path = (String) MiniJson.obj(o).get("path");
                File target = new File(dest, path);
                File parent = target.getParentFile();
                if (parent != null && !parent.isDirectory() && !parent.mkdirs()) throw new IOException("mkdir " + parent);
                //noinspection ResultOfMethodCallIgnored
                target.delete();
                if (!new File(tmp, path).renameTo(target)) throw new IOException("rename failed: " + path);
            }
            deleteRecursive(tmp);
        } else {
            try (OutputStream os = new FileOutputStream(new File(tmp, "pack.json"))) {
                os.write(packJson.getBytes(StandardCharsets.UTF_8));
            }
            deleteRecursive(dest);
            if (!tmp.renameTo(dest)) throw new IOException("rename failed");
        }

        Report rep = new Report();
        rep.files = files.size();
        rep.bytes = done;
        rep.millis = System.currentTimeMillis() - t0;
        return rep;
    }

    /** display name -> document id for the direct children of a directory document. */
    private Map<String, String> list(String parentId) {
        Map<String, String> m = new HashMap<>();
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId);
        try (Cursor c = cr.query(children, new String[]{
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME}, null, null, null)) {
            while (c != null && c.moveToNext()) m.put(c.getString(1), c.getString(0));
        }
        return m;
    }

    private String readText(String docId) throws IOException {
        try (InputStream in = cr.openInputStream(DocumentsContract.buildDocumentUriUsingTree(tree, docId))) {
            if (in == null) throw new IOException("cannot open pack.json");
            byte[] all = in.readAllBytes();
            if (all.length > (1 << 20)) throw new IOException("pack.json too large");
            return new String(all, StandardCharsets.UTF_8);
        }
    }

    static void deleteRecursive(File f) {
        if (!f.exists()) return;
        File[] kids = f.listFiles();
        if (kids != null) for (File k : kids) deleteRecursive(k);
        //noinspection ResultOfMethodCallIgnored
        f.delete();
    }

    private static String hex(byte[] b) {
        StringBuilder s = new StringBuilder();
        for (byte x : b) s.append(String.format("%02x", x & 0xff));
        return s.toString();
    }
}
