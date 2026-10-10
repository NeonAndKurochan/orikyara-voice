package local.fereko.probe;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.view.View;

import java.util.function.DoubleSupplier;

/**
 * ふぇれ子's stand picture (one cut-out still until the 3D model is ready), drawn over the
 * background: picture rows srcTop..srcBottom (bitmap pixels, e.g. ears to knees) fill the view
 * from viewTop to viewBottom (fractions of its height), centred on picture column centerX. It breathes slowly, and while she
 * talks it bobs with the loudness of her voice, so it is clear who is speaking.
 */
final class StandView extends View {
    private static final double BREATH_PERIOD_S = 3.6, BREATH_SCALE = 0.005;
    private static final float BOB_DP = 6;

    private final Bitmap bmp;
    private final DoubleSupplier level;
    private final float srcTop, srcBottom, viewTop, viewBottom, centerX;
    private final Matrix m = new Matrix();
    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG | Paint.ANTI_ALIAS_FLAG);
    private final long t0 = System.nanoTime();
    private double smooth;

    /** @param level loudness of the voice being heard now (0..1) */
    StandView(Context c, Bitmap bmp, DoubleSupplier level, float srcTop, float srcBottom, float viewTop, float viewBottom, float centerX) {
        super(c);
        this.bmp = bmp;
        this.level = level;
        this.srcTop = srcTop;
        this.srcBottom = srcBottom;
        this.viewTop = viewTop;
        this.viewBottom = viewBottom;
        this.centerX = centerX;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        int w = getWidth(), h = getHeight();
        if (w == 0 || h == 0) return;
        float scale = (viewBottom - viewTop) * h / (srcBottom - srcTop);
        float dp = getResources().getDisplayMetrics().density;
        double t = (System.nanoTime() - t0) / 1e9;
        // a speech level of ~0.1 RMS is a normal voice; smoothed so the motion isn't jittery
        double target = Math.min(1, level.getAsDouble() / 0.1);
        smooth += (target - smooth) * (target > smooth ? 0.5 : 0.15);
        float breath = (float) (1 + BREATH_SCALE * Math.sin(2 * Math.PI * t / BREATH_PERIOD_S));
        float bob = (float) (smooth * BOB_DP * dp);
        m.setScale(scale, scale);
        m.postScale(1, breath, 0, bmp.getHeight() * scale);   // breathe from the feet up
        m.postTranslate(w / 2f - centerX * scale, viewTop * h - srcTop * scale - bob);
        canvas.drawBitmap(bmp, m, paint);
        postInvalidateOnAnimation();
    }
}
