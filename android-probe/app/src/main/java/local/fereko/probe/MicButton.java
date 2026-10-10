package local.fereko.probe;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

/**
 * Round microphone button drawn in code (white on dark glass); while listening it turns red
 * (ふぇれ子's eye colour) with a slowly pulsing ring.
 */
final class MicButton extends View {
    static final int ACCENT = 0xFFE5484D;

    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG), line = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF r = new RectF();
    private boolean listening;
    private final long t0 = System.nanoTime();

    MicButton(Context c) {
        super(c);
        line.setStyle(Paint.Style.STROKE);
        line.setStrokeCap(Paint.Cap.ROUND);
        setClickable(true);
        setContentDescription("マイク");
    }

    void setListening(boolean on) {
        listening = on;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas c) {
        float w = getWidth(), h = getHeight(), cx = w / 2, cy = h / 2, rad = Math.min(w, h) / 2;
        float dp = getResources().getDisplayMetrics().density;
        if (listening) {
            double t = (System.nanoTime() - t0) / 1e9;
            float pulse = (float) (0.5 + 0.5 * Math.sin(t * 2 * Math.PI / 1.2));
            fill.setColor(ACCENT);
            fill.setAlpha((int) (60 + 60 * pulse));
            c.drawCircle(cx, cy, rad, fill);
            fill.setAlpha(255);
            c.drawCircle(cx, cy, rad - 5 * dp, fill);
            postInvalidateOnAnimation();
        } else {
            fill.setColor(0x66000000);
            c.drawCircle(cx, cy, rad, fill);
            line.setColor(0x80FFFFFF);
            line.setStrokeWidth(1.2f * dp);
            c.drawCircle(cx, cy, rad - dp, line);
        }
        // microphone: capsule, cradle arc and stem
        line.setColor(0xFFFFFFFF);
        line.setStrokeWidth(2.2f * dp);
        // a tall narrow capsule (a round one read as a person icon), a cradle under it, stem and foot
        float cw = rad * 0.16f, top = cy - rad * 0.52f, bottom = cy + rad * 0.12f;
        r.set(cx - cw, top, cx + cw, bottom);
        fill.setColor(0xFFFFFFFF);
        c.drawRoundRect(r, cw, cw, fill);
        float aw = rad * 0.32f;
        r.set(cx - aw, bottom - rad * 0.36f, cx + aw, bottom + rad * 0.2f);
        c.drawArc(r, 0, 180, false, line);
        float stemTop = bottom + rad * 0.2f, stemBottom = cy + rad * 0.5f;
        c.drawLine(cx, stemTop, cx, stemBottom, line);
        c.drawLine(cx - rad * 0.18f, stemBottom, cx + rad * 0.18f, stemBottom, line);
    }
}
