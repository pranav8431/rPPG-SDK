"""One-shot Heart Rate & Blood Pressure measurement.

Accumulates ~10 seconds of data for a single high-accuracy reading,
then displays the final result and stops.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from sdk.rppg_sdk import RPPGSDK
from sdk.heart_rate import HeartRate
from sdk.vitals import BloodPressureEstimator

# Frames to accumulate before measurement (~10 seconds at 30 fps)
REQUIRED_FRAMES = 300
# Maximum frames before giving up if vitals aren't ready
MAX_FRAMES = 600

# UI
TARGET_W = 640
BG_COLOR = (30, 30, 30)
ACCENT = (206, 255, 59)
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)
DARK_GRAY = (80, 80, 80)
GREEN = (100, 220, 120)

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]


def draw_face_contour(frame, landmarks):
    """Draw subtle face contour ellipse."""
    if landmarks is None:
        return
    pts = [landmarks[i] for i in FACE_OVAL if i < len(landmarks)]
    if len(pts) < 5:
        return
    pts = np.array(pts, dtype=np.int32)
    ellipse = cv2.fitEllipse(pts)
    cv2.ellipse(frame, ellipse, ACCENT, 2, cv2.LINE_AA)


def draw_progress(frame, pct, h, w):
    """Draw measurement progress bar and text."""
    bar_w = int(w * 0.6)
    bar_h = 18
    x0 = (w - bar_w) // 2
    y0 = h - 55

    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), DARK_GRAY, -1)
    fill_w = int(bar_w * pct / 100)
    if fill_w > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), ACCENT, -1)

    text = f"Measuring... {int(pct)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, 0.55, 1)
    cv2.putText(frame, text, ((w - tw) // 2, y0 - 10),
                font, 0.55, WHITE, 1, cv2.LINE_AA)


def build_result_image(bpm, bp_sys, bp_dia, width=TARGET_W, height=380):
    """Create a static result frame showing HR and BP."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = BG_COLOR
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    title = "Measurement Complete"
    (tw, _), _ = cv2.getTextSize(title, font, 0.85, 2)
    cv2.putText(img, title, ((width - tw) // 2, 50),
                font, 0.85, GREEN, 2, cv2.LINE_AA)

    # Divider
    cv2.line(img, (width // 4, 70), (3 * width // 4, 70), DARK_GRAY, 1)

    # Heart Rate
    hr_label = "Heart Rate"
    hr_val = f"{bpm:.0f}" if bpm else "--"
    hr_unit = "bpm"
    y = 130
    (lw, _), _ = cv2.getTextSize(hr_label, font, 0.55, 1)
    cv2.putText(img, hr_label, ((width - lw) // 2, y),
                font, 0.55, GRAY, 1, cv2.LINE_AA)
    (vw, vh), _ = cv2.getTextSize(hr_val, font, 1.8, 3)
    vx = (width - vw) // 2 - 20
    cv2.putText(img, hr_val, (vx, y + 55),
                font, 1.8, WHITE, 3, cv2.LINE_AA)
    cv2.putText(img, hr_unit, (vx + vw + 8, y + 55),
                font, 0.55, GRAY, 1, cv2.LINE_AA)

    # Blood Pressure
    bp_label = "Blood Pressure"
    bp_val = f"{bp_sys:.0f}/{bp_dia:.0f}" if bp_sys and bp_dia else "--"
    bp_unit = "mmHg"
    y = 250
    (lw, _), _ = cv2.getTextSize(bp_label, font, 0.55, 1)
    cv2.putText(img, bp_label, ((width - lw) // 2, y),
                font, 0.55, GRAY, 1, cv2.LINE_AA)
    (vw, vh), _ = cv2.getTextSize(bp_val, font, 1.8, 3)
    vx = (width - vw) // 2 - 25
    cv2.putText(img, bp_val, (vx, y + 55),
                font, 1.8, WHITE, 3, cv2.LINE_AA)
    cv2.putText(img, bp_unit, (vx + vw + 8, y + 55),
                font, 0.55, GRAY, 1, cv2.LINE_AA)

    # Footer
    quit_text = "Press 'q' to exit"
    (qw, _), _ = cv2.getTextSize(quit_text, font, 0.45, 1)
    cv2.putText(img, quit_text, ((width - qw) // 2, height - 20),
                font, 0.45, DARK_GRAY, 1, cv2.LINE_AA)

    return img


def main():
    print("One-shot BP & Heart Rate measurement.")
    print("Please sit still and look at the camera.")
    print("Press 'q' to quit at any time.\n")

    sdk = RPPGSDK(source=0, buffer_size=512)
    # Use larger smoothing windows for one-shot accuracy
    sdk.heart_rate = HeartRate(smooth_window=15)
    sdk.bp_estimator = BloodPressureEstimator(smooth_window=15)

    measuring = True

    try:
        while True:
            if measuring:
                frame, vitals = sdk.run()
                if frame is None:
                    print("Failed to read frame. Exiting.")
                    break

                h, w = frame.shape[:2]
                if w != TARGET_W:
                    scale = TARGET_W / w
                    frame = cv2.resize(frame, (TARGET_W, int(h * scale)))
                    h, w = frame.shape[:2]

                # Scale landmarks for resized frame
                landmarks = sdk.last_landmarks
                if landmarks is not None and w != frame.shape[1]:
                    s = TARGET_W / frame.shape[1]
                    landmarks = [(int(x * s), int(y * s)) for x, y in landmarks]

                draw_face_contour(frame, landmarks)

                buf_len = sdk.signal_extractor.length
                progress = min(99, int(buf_len / REQUIRED_FRAMES * 100))

                bpm = vitals.get('bpm')
                bp_sys = vitals.get('bp_sys')
                bp_dia = vitals.get('bp_dia')

                ready = (buf_len >= REQUIRED_FRAMES
                         and bpm is not None
                         and bp_sys is not None
                         and bp_dia is not None)
                timed_out = buf_len >= MAX_FRAMES

                if ready or timed_out:
                    measuring = False
                    sdk.release()

                    print("=" * 40)
                    print("  MEASUREMENT COMPLETE")
                    print("=" * 40)
                    if bpm:
                        print(f"  Heart Rate:     {bpm:.0f} bpm")
                    else:
                        print("  Heart Rate:     --")
                    if bp_sys and bp_dia:
                        print(f"  Blood Pressure: {bp_sys:.0f}/{bp_dia:.0f} mmHg")
                    else:
                        print("  Blood Pressure: --")
                    print("=" * 40)

                    result = build_result_image(bpm, bp_sys, bp_dia)
                    cv2.imshow("BP & Heart Rate", result)
                else:
                    draw_progress(frame, progress, h, w)
                    cv2.imshow("BP & Heart Rate", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if measuring:
            sdk.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
