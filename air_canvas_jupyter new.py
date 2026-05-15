"""
╔══════════════════════════════════════════════════════════════════╗
║          AIR CANVAS — Jupyter Notebook Edition                   ║
║   Draw in the air using a highlighted pen cap tracked by camera  ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE:
1. Run this cell in Jupyter Notebook
2. Hold a pen/marker with a BRIGHTLY COLORED cap toward camera
   → Best colors: Bright GREEN, ORANGE, RED, PINK, or BLUE cap
3. The tracker will lock onto the dominant color in a 5-second calibration
4. Draw freely in the air! The tip is tracked with sub-pixel accuracy.

REQUIREMENTS:
    pip install opencv-python numpy ipywidgets IPython

CONTROLS (keyboard, click the output cell first):
    C  → Calibrate / re-pick color
    E  → Erase / clear canvas
    S  → Save canvas as PNG
    Q  → Quit
    1-5 → Switch brush color
    [ / ] → Decrease / Increase brush size
"""

import cv2
import numpy as np
import threading
import time
import os
from collections import deque
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from IPython import get_ipython

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
class Config:
    # Tracking
    CALIB_SECONDS      = 4        # seconds to auto-calibrate color
    CALIB_BOX_FRAC     = 0.12     # fraction of frame for calib ROI box
    HSV_H_TOLERANCE    = 15       # hue tolerance (±)
    HSV_S_MIN          = 80       # min saturation to keep
    HSV_V_MIN          = 80       # min value/brightness to keep
    MIN_CONTOUR_AREA   = 150      # ignore blobs smaller than this
    SMOOTH_WINDOW      = 5        # Kalman / EMA window size
    
    # Canvas
    CANVAS_ALPHA       = 0.85     # canvas overlay opacity (0=invisible,1=opaque)
    BRUSH_SIZE_DEFAULT = 6
    BRUSH_SIZES        = [3, 5, 8, 12, 18]
    
    # Brush palette
    COLORS = {
        "1 Red"    : (0,   0,   220),
        "2 Blue"   : (220, 80,  0  ),
        "3 Green"  : (30,  180, 30 ),
        "4 Yellow" : (0,   220, 220),
        "5 White"  : (255, 255, 255),
    }
    ERASER_COLOR = (0, 0, 0)
    
    # Display
    WIN_NAME = "Air Canvas"
    FPS_TARGET = 30


# ─────────────────────────────────────────────────────────────────
#  KALMAN SMOOTHER  (x, y position)
# ─────────────────────────────────────────────────────────────────
class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost       = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized = True
        self.kf.correct(meas)
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])

    def predict_only(self):
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])

    def reset(self):
        self.initialized = False


# ─────────────────────────────────────────────────────────────────
#  COLOR CALIBRATOR
# ─────────────────────────────────────────────────────────────────
class ColorCalibrator:
    def __init__(self):
        self.lower = None
        self.upper = None
        self.calibrated = False
        self._samples = []

    def add_sample(self, roi_bgr):
        """Feed a BGR ROI patch during calibration phase."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        self._samples.append(hsv)

    def finalize(self):
        if not self._samples:
            return False
        all_pixels = np.concatenate([s.reshape(-1, 3) for s in self._samples])
        # Use median hue; robust to outliers
        h_med = np.median(all_pixels[:, 0])
        s_med = np.median(all_pixels[:, 1])
        v_med = np.median(all_pixels[:, 2])
        ht = Config.HSV_H_TOLERANCE
        self.lower = np.array([max(0, h_med-ht), Config.HSV_S_MIN, Config.HSV_V_MIN], np.uint8)
        self.upper = np.array([min(179, h_med+ht), 255, 255], np.uint8)
        self.calibrated = True
        self._samples = []
        return True

    def mask(self, frame_bgr):
        """Return binary mask for calibrated color."""
        if not self.calibrated:
            return None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lower, self.upper)
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel, iterations=2)
        return m

    def reset(self):
        self.lower = None
        self.upper = None
        self.calibrated = False
        self._samples = []


# ─────────────────────────────────────────────────────────────────
#  STROKE RENDERER  (anti-aliased thick lines with pressure feel)
# ─────────────────────────────────────────────────────────────────
class StrokeRenderer:
    def __init__(self, canvas_shape):
        h, w = canvas_shape[:2]
        self.canvas = np.zeros((h, w, 3), np.uint8)
        self.prev_pt = None

    def draw(self, pt, color_bgr, size):
        if self.prev_pt is not None and pt is not None:
            # Draw thick anti-aliased line
            cv2.line(self.canvas, self.prev_pt, pt, color_bgr, size, cv2.LINE_AA)
            # Add a filled circle at the joint for smoothness
            cv2.circle(self.canvas, pt, size // 2, color_bgr, -1, cv2.LINE_AA)
        self.prev_pt = pt

    def lift(self):
        self.prev_pt = None

    def clear(self):
        self.canvas[:] = 0
        self.prev_pt = None


# ─────────────────────────────────────────────────────────────────
#  MAIN AIR CANVAS APP
# ─────────────────────────────────────────────────────────────────
class AirCanvas:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("❌ Could not open camera. Check camera_index.")
        
        self.h, self.w = frame.shape[:2]
        self.calibrator  = ColorCalibrator()
        self.kalman      = KalmanTracker()
        self.renderer    = StrokeRenderer((self.h, self.w))
        
        # State
        self.state       = "calibrating"   # calibrating | tracking
        self.calib_start = time.time()
        self.pen_down    = True            # always drawing when tip detected
        self.lost_frames = 0
        self.MAX_LOST    = 8               # frames before lifting pen
        
        # Brush
        color_items      = list(Config.COLORS.items())
        self.brush_color = color_items[0][1]
        self.brush_idx   = 0
        self.brush_size  = Config.BRUSH_SIZE_DEFAULT
        self.color_names = color_items
        
        # Trail deque for ghost trail effect
        self.trail       = deque(maxlen=20)
        
        # FPS
        self._fps_t      = time.time()
        self._fps_count  = 0
        self._fps        = 0.0

    # ── helpers ───────────────────────────────────────────────────
    def _calib_roi(self, frame):
        """Return the center calibration ROI."""
        bx = int(self.w * Config.CALIB_BOX_FRAC)
        by = int(self.h * Config.CALIB_BOX_FRAC)
        cx, cy = self.w // 2, self.h // 2
        return frame[cy-by:cy+by, cx-bx:cx+bx], (cx-bx, cy-by, cx+bx, cy+by)

    def _find_tip(self, mask):
        """
        Find the topmost point of the largest contour = pen tip.
        Returns (x, y) or None.
        """
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        # Pick largest by area
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < Config.MIN_CONTOUR_AREA:
            return None
        # Topmost point = pen tip (user points pen up at camera)
        tip = tuple(c[c[:, :, 1].argmin()][0])
        return tip

    def _draw_ui(self, frame, tip_raw, tip_smooth):
        """Overlay HUD, palette, status on the live frame."""
        overlay = frame.copy()
        
        # ── Palette bar ──────────────────────────────────────────
        bar_h = 50
        bar_y = self.h - bar_h - 10
        for i, (name, bgr) in enumerate(self.color_names):
            x = 10 + i * 60
            selected = (i == self.brush_idx)
            cv2.rectangle(overlay, (x, bar_y), (x+50, bar_y+bar_h), bgr, -1)
            if selected:
                cv2.rectangle(overlay, (x-3, bar_y-3), (x+53, bar_y+bar_h+3), (255,255,255), 3)
            cv2.putText(overlay, name[0], (x+18, bar_y+33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        # ── Brush size indicator ──────────────────────────────────
        cv2.circle(overlay, (self.w - 60, self.h - 60), self.brush_size, self.brush_color, -1, cv2.LINE_AA)
        cv2.putText(overlay, f"sz:{self.brush_size}", (self.w-115, self.h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        # ── FPS ───────────────────────────────────────────────────
        cv2.putText(overlay, f"FPS: {self._fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100), 2)

        # ── State label ───────────────────────────────────────────
        state_txt = "CALIBRATING — hold pen cap in box" if self.state == "calibrating" else "DRAWING"
        cv2.putText(overlay, state_txt, (self.w//2 - 180, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,255), 2)

        # ── Calibration box ──────────────────────────────────────
        if self.state == "calibrating":
            _, (x1, y1, x2, y2) = self._calib_roi(frame)
            elapsed = time.time() - self.calib_start
            progress = min(elapsed / Config.CALIB_SECONDS, 1.0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
            # Progress arc
            bar_len = int((x2 - x1) * progress)
            cv2.rectangle(overlay, (x1, y2+5), (x1+bar_len, y2+12), (0,255,100), -1)

        # ── Tip crosshair ─────────────────────────────────────────
        if tip_smooth:
            x, y = tip_smooth
            # Ghost trail
            for j, pt in enumerate(self.trail):
                alpha_t = j / max(len(self.trail), 1)
                r = max(1, int(self.brush_size * alpha_t * 0.6))
                cv2.circle(overlay, pt, r, self.brush_color, -1, cv2.LINE_AA)
            # Crosshair
            arm = 18
            cv2.line(overlay, (x-arm, y), (x+arm, y), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(overlay, (x, y-arm), (x, y+arm), (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), self.brush_size+4, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), self.brush_size,   self.brush_color, -1, cv2.LINE_AA)

        # ── Controls cheat-sheet ─────────────────────────────────
        help_lines = ["C=calib  E=erase  S=save  Q=quit", "1-5=color  [/]=size"]
        for i, l in enumerate(help_lines):
            cv2.putText(overlay, l, (10, self.h - 100 - i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1)

        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    def _update_fps(self):
        self._fps_count += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._fps = self._fps_count / (now - self._fps_t)
            self._fps_count = 0
            self._fps_t = now

    # ── main loop ─────────────────────────────────────────────────
    def run(self):
        print("═" * 60)
        print("  AIR CANVAS  |  Jupyter Edition")
        print("  Hold pen cap in the CENTER BOX to calibrate color")
        print("  Keys: C=recalibrate  E=erase  S=save  Q=quit")
        print("        1-5=color   [=smaller  ]=larger brush")
        print("═" * 60)

        cv2.namedWindow(Config.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(Config.WIN_NAME, min(self.w, 1280), min(self.h, 720))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # mirror for natural drawing
            self._update_fps()

            tip_raw    = None
            tip_smooth = None

            # ── CALIBRATION PHASE ─────────────────────────────────
            if self.state == "calibrating":
                roi, _ = self._calib_roi(frame)
                self.calibrator.add_sample(roi)
                elapsed = time.time() - self.calib_start
                if elapsed >= Config.CALIB_SECONDS:
                    ok = self.calibrator.finalize()
                    if ok:
                        self.state = "tracking"
                        self.kalman.reset()
                        print(f"✅ Calibrated! HSV range: {self.calibrator.lower} – {self.calibrator.upper}")
                    else:
                        print("⚠️  Calibration failed, retrying…")
                        self.calib_start = time.time()
                        self.calibrator.reset()

            # ── TRACKING PHASE ────────────────────────────────────
            elif self.state == "tracking":
                mask = self.calibrator.mask(frame)
                if mask is not None:
                    tip_raw = self._find_tip(mask)
                    if tip_raw:
                        self.lost_frames = 0
                        tip_smooth = self.kalman.update(*tip_raw)
                        self.trail.append(tip_smooth)
                        self.renderer.draw(tip_smooth, self.brush_color, self.brush_size)
                    else:
                        self.lost_frames += 1
                        if self.lost_frames <= self.MAX_LOST:
                            tip_smooth = self.kalman.predict_only()
                        else:
                            self.renderer.lift()
                            self.trail.clear()

            # ── COMPOSE OUTPUT ────────────────────────────────────
            # Blend canvas onto live frame
            canvas_gray = cv2.cvtColor(self.renderer.canvas, cv2.COLOR_BGR2GRAY)
            canvas_mask = canvas_gray > 0
            display_frame = frame.copy()
            display_frame[canvas_mask] = (
                display_frame[canvas_mask] * (1 - Config.CANVAS_ALPHA) +
                self.renderer.canvas[canvas_mask] * Config.CANVAS_ALPHA
            ).astype(np.uint8)

            self._draw_ui(display_frame, tip_raw, tip_smooth)
            cv2.imshow(Config.WIN_NAME, display_frame)

            # ── KEY HANDLING ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('e'):
                self.renderer.clear()
                self.trail.clear()
                print("🧹 Canvas cleared")
            elif key == ord('c'):
                self.state = "calibrating"
                self.calib_start = time.time()
                self.calibrator.reset()
                self.kalman.reset()
                print("🎯 Recalibrating…")
            elif key == ord('s'):
                fname = f"air_canvas_{int(time.time())}.png"
                cv2.imwrite(fname, self.renderer.canvas)
                print(f"💾 Saved → {fname}")
            elif ord('1') <= key <= ord('5'):
                idx = key - ord('1')
                if idx < len(self.color_names):
                    self.brush_idx = idx
                    self.brush_color = self.color_names[idx][1]
                    print(f"🖌  Color: {self.color_names[idx][0]}")
            elif key == ord('['):
                self.brush_size = max(2, self.brush_size - 2)
            elif key == ord(']'):
                self.brush_size = min(30, self.brush_size + 2)

        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Air Canvas closed.")


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def launch(camera_index=0):
    """
    Launch the Air Canvas.
    
    Args:
        camera_index: OpenCV camera index (default 0 = built-in webcam).
                      Try 1 or 2 if you have multiple cameras.
    """
    try:
        app = AirCanvas(camera_index=camera_index)
        app.run()
    except RuntimeError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n⏹  Interrupted by user.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    launch()
