import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from sdk.rppg_sdk import RPPGSDK

# MediaPipe FaceMesh face oval indices — correct sequential order around the jawline
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]

# UI Colors (BGR)
CYAN = (206, 255, 59)
BAR_BG = (35, 30, 25)
CARD_BG = (50, 44, 38)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
MID_GRAY = (140, 140, 140)
DARK_GRAY = (80, 80, 80)
DASH_GRAY = (100, 100, 100)
COLOR_HR = (100, 80, 255)       # coral/red
COLOR_BP = (255, 170, 120)     # light blue
COLOR_SPO2 = (200, 150, 50)    # blue/teal
COLOR_RR = (120, 220, 100)     # green

# Bar dimensions
BAR_H = 100
CARD_PAD = 6
CARD_RAD = 8

# Target window width for a clean layout
TARGET_W = 800


def draw_face_overlay(frame, landmarks):
    """Draw face contour with smooth elliptical overlay."""
    if landmarks is None:
        return
    h, w = frame.shape[:2]

    # Gather valid oval points
    pts = [landmarks[i] for i in FACE_OVAL if i < len(landmarks)]
    if len(pts) < 10:
        return
    pts = np.array(pts, dtype=np.int32)

    # Fit an ellipse for a smooth contour (needs >= 5 points)
    if len(pts) >= 5:
        ellipse = cv2.fitEllipse(pts)
        overlay = frame.copy()
        cv2.ellipse(overlay, ellipse, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.ellipse(frame, ellipse, CYAN, 3, cv2.LINE_AA)
    else:
        hull = cv2.convexHull(pts)
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, hull, (255, 255, 255))
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [hull], True, CYAN, 3, cv2.LINE_AA)


def draw_motion_text(frame, level, h, w):
    """Draw 'Motion' indicator when movement is detected."""
    if level < 5:
        return
    text = "Motion"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, 1.2 * w / 640)
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = (w - tw) // 2, int(h * 0.72)
    overlay = frame.copy()
    cv2.putText(overlay, text, (x, y), font, scale, LIGHT_GRAY, thick, cv2.LINE_AA)
    alpha = min(0.6, level / 20.0)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_progress_circle(frame, pct, cx, cy, r=28):
    """Draw calibration progress arc with percentage."""
    cv2.circle(frame, (cx, cy), r, DARK_GRAY, 2, cv2.LINE_AA)
    angle = int(360 * pct / 100)
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, angle, CYAN, 3, cv2.LINE_AA)
    text = f"{int(pct)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                font, 0.45, CYAN, 1, cv2.LINE_AA)


def _rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Draw a rounded rectangle."""
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness, cv2.LINE_AA)


# --- Icon drawing ---

def _draw_heart(img, cx, cy, s, c):
    r = max(s // 2, 1)
    cv2.circle(img, (cx - r + 1, cy - r // 2), r, c, -1, cv2.LINE_AA)
    cv2.circle(img, (cx + r - 1, cy - r // 2), r, c, -1, cv2.LINE_AA)
    tri = np.array([[cx - s, cy], [cx, cy + s], [cx + s, cy]], np.int32)
    cv2.fillConvexPoly(img, tri, c, cv2.LINE_AA)


def _draw_drop(img, cx, cy, s, c):
    cv2.circle(img, (cx, cy + s // 3), s // 2, c, -1, cv2.LINE_AA)
    tri = np.array([[cx, cy - s], [cx - s // 2, cy + s // 4],
                    [cx + s // 2, cy + s // 4]], np.int32)
    cv2.fillConvexPoly(img, tri, c, cv2.LINE_AA)


def _draw_ring(img, cx, cy, s, c):
    cv2.circle(img, (cx, cy), s, c, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(s // 3, 1), c, -1, cv2.LINE_AA)


def _draw_fan(img, cx, cy, s, c):
    for a in range(0, 360, 90):
        rad = np.radians(a)
        x2, y2 = int(cx + s * np.cos(rad)), int(cy + s * np.sin(rad))
        cv2.line(img, (cx, cy), (x2, y2), c, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(s // 3, 1), c, -1, cv2.LINE_AA)


def draw_vitals_bar(canvas, vitals, bar_y, w):
    """Draw the bottom dashboard bar with 5 spaced-out vital metric cards."""
    # Full dark background
    cv2.rectangle(canvas, (0, bar_y), (w, bar_y + BAR_H), BAR_BG, -1)

    bpm = vitals.get('bpm')
    spo2 = vitals.get('spo2')
    bp_sys = vitals.get('bp_sys')
    bp_dia = vitals.get('bp_dia')
    rr = vitals.get('resp_rate')

    bp_val = f"{bp_sys:.0f}/{bp_dia:.0f}" if bp_sys and bp_dia else "--"

    metrics = [
        ("Heart Rate",     f"{bpm:.0f}" if bpm else "--",  "bpm",   COLOR_HR,   _draw_heart),
        ("Blood Pressure", bp_val,                          "mmHg",  COLOR_BP,   _draw_drop),
        ("SpO2",           f"{spo2:.0f}" if spo2 else "--", "%",     COLOR_SPO2, _draw_ring),
        ("Resp Rate",      f"{rr:.0f}" if rr else "--",     "bpm",   COLOR_RR,   _draw_fan),
    ]

    n = len(metrics)
    total_pad = CARD_PAD * (n + 1)
    card_w = (w - total_pad) // n
    card_h = BAR_H - CARD_PAD * 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (label, val, unit, accent, icon_fn) in enumerate(metrics):
        x0 = CARD_PAD + i * (card_w + CARD_PAD)
        y0 = bar_y + CARD_PAD
        x1 = x0 + card_w
        y1 = y0 + card_h

        # Card background
        _rounded_rect(canvas, x0, y0, x1, y1, CARD_RAD, CARD_BG)

        # Accent top line
        cv2.line(canvas, (x0 + CARD_RAD, y0), (x1 - CARD_RAD, y0), accent, 2)

        cx = (x0 + x1) // 2

        # Label (top, small, gray)
        (lw, lh), _ = cv2.getTextSize(label, font, 0.38, 1)
        cv2.putText(canvas, label, (cx - lw // 2, y0 + 18),
                    font, 0.38, MID_GRAY, 1, cv2.LINE_AA)

        # Value (center, large, white or gray if --)
        is_active = val not in ("--",)
        val_color = WHITE if is_active else DASH_GRAY
        val_scale = 0.75 if len(val) <= 3 else 0.6
        (vw, vh), _ = cv2.getTextSize(val, font, val_scale, 2)
        cv2.putText(canvas, val, (cx - vw // 2, y0 + 20 + vh + 10),
                    font, val_scale, val_color, 2, cv2.LINE_AA)

        # Unit (small, beside value)
        (uw, uh), _ = cv2.getTextSize(unit, font, 0.32, 1)
        cv2.putText(canvas, unit, (cx + vw // 2 + 4, y0 + 20 + vh + 10),
                    font, 0.32, MID_GRAY, 1, cv2.LINE_AA)

        # Icon (bottom right of card)
        icon_fn(canvas, x1 - 16, y1 - 14, 8, accent)


def main():
    """Run real-time vital signs monitoring from webcam."""
    sdk = RPPGSDK(source=0)
    min_frames = sdk.MIN_FRAMES
    print("Starting rPPG vital signs monitor. Press 'q' to quit.")

    prev_gray = None
    motion_level = 0.0

    try:
        while True:
            frame, vitals = sdk.run()
            if frame is None:
                print("Failed to read frame. Exiting.")
                break

            h, w = frame.shape[:2]

            # Resize for consistent UI if too small
            orig_w = w
            if w < TARGET_W:
                scale = TARGET_W / w
                frame = cv2.resize(frame, (TARGET_W, int(h * scale)))
                h, w = frame.shape[:2]
            else:
                scale = 1.0

            # Scale landmarks to match resized frame
            landmarks = sdk.last_landmarks
            if landmarks is not None and scale != 1.0:
                landmarks = [(int(x * scale), int(y * scale)) for x, y in landmarks]

            # Motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                motion_level = (0.7 * motion_level
                                + 0.3 * float(np.mean(cv2.absdiff(gray, prev_gray))))
            prev_gray = gray.copy()

            # Face overlay
            draw_face_overlay(frame, landmarks)

            # Motion indicator
            draw_motion_text(frame, motion_level, h, w)

            # Calibration progress circle
            buf_len = sdk.signal_extractor.length
            progress = min(100, int(buf_len / min_frames * 100))
            if progress < 100:
                draw_progress_circle(frame, progress, w - 45, h - 45)

            # Build canvas: camera frame + vitals bar
            canvas = np.zeros((h + BAR_H, w, 3), dtype=np.uint8)
            canvas[:h, :w] = frame
            draw_vitals_bar(canvas, vitals, h, w)

            cv2.imshow("rPPG Vital Signs Monitor", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        sdk.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
