# rPPG SDK — Remote Photoplethysmography Vital Signs Monitor

A real-time, non-contact vital signs monitoring SDK that extracts physiological measurements from a standard webcam using remote photoplethysmography (rPPG).

---

## Features

- **Heart Rate (BPM)** — Dual spectral + time-domain estimation with parabolic interpolation for sub-BPM accuracy
- **Blood Pressure (mmHg)** — Systolic/diastolic estimation via pulse wave morphology analysis
- **Blood Oxygen Saturation (SpO2 %)** — Ratio-of-ratios technique on RGB channel pulsatile components
- **Respiration Rate (breaths/min)** — Hilbert transform envelope extraction with spectral peak analysis
- **Face Detection & Tracking** — MediaPipe FaceLandmarker with 468 landmarks for precise ROI extraction
- **Motion Detection** — Frame-to-frame motion indicator for signal quality awareness
- **Real-time Dashboard** — Live webcam feed with face overlay and vital signs display

---

## Installation

```bash
# Clone the repository
git clone https://github.com/pranav8431/rPPG-SDK
cd rPPG-SDK

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| opencv-python | >= 4.5 |
| mediapipe | >= 0.10 |
| numpy | >= 1.21 |
| scipy | >= 1.7 |

---

## Run the Demo

The quickest way to see everything working:

```bash
python examples/webcam_demo.py
```

This opens a window with:
- Live webcam feed
- Green face contour overlay
- Calibration progress circle (bottom-right, fills over ~2 seconds)
- Motion indicator text when movement is detected
- Bottom dashboard showing **Heart Rate**, **Blood Pressure**, **SpO2**, and **Respiration Rate**

Press **q** to quit.

---

## Using the SDK

### Basic Usage — get vitals every frame

```python
from sdk.rppg_sdk import RPPGSDK

sdk = RPPGSDK(source=0)  # source=0 is the default webcam

try:
    while True:
        frame, vitals = sdk.run()

        if frame is None:
            break  # camera disconnected

        bpm       = vitals['bpm']        # float or None
        spo2      = vitals['spo2']       # float or None  (%)
        bp_sys    = vitals['bp_sys']     # float or None  (mmHg)
        bp_dia    = vitals['bp_dia']     # float or None  (mmHg)
        resp_rate = vitals['resp_rate']  # float or None  (breaths/min)

        if bpm is not None:
            print(f"HR: {bpm:.0f} bpm | SpO2: {spo2:.0f}% | "
                  f"BP: {bp_sys:.0f}/{bp_dia:.0f} mmHg | RR: {resp_rate:.0f} bpm")
        else:
            print("Calibrating...")
finally:
    sdk.release()
```

> Vitals return `None` while the buffer is still filling (first ~2 seconds). Always check before using a value.

---

### Display the frame with OpenCV

```python
import cv2
from sdk.rppg_sdk import RPPGSDK

sdk = RPPGSDK(source=0)

try:
    while True:
        frame, vitals = sdk.run()
        if frame is None:
            break

        bpm = vitals['bpm']
        label = f"HR: {bpm:.0f} bpm" if bpm else "Calibrating..."
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("rPPG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    sdk.release()
    cv2.destroyAllWindows()
```

---

### Use a video file instead of a webcam

```python
sdk = RPPGSDK(source="path/to/video.mp4")
```

---

### Configuration options

```python
sdk = RPPGSDK(
    source=0,          # int (camera index) or str (video file path)
    buffer_size=256,   # how many frames to keep in the signal ring buffer
    window_size=32,    # POS algorithm sliding window length (frames)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | `0` | Camera index or video file path |
| `buffer_size` | `256` | Signal buffer length in frames (~8 sec at 30 fps) |
| `window_size` | `32` | POS algorithm window (~1 sec at 30 fps) |

---

### Access the detected face landmarks

After each `sdk.run()` call, `sdk.last_landmarks` holds a list of 468 `(x, y)` pixel coordinates (or `None` if no face was found):

```python
frame, vitals = sdk.run()
landmarks = sdk.last_landmarks  # list of (x, y) tuples, or None

if landmarks is not None:
    # e.g. draw a point at the nose tip (landmark 1)
    import cv2
    cv2.circle(frame, landmarks[1], 4, (0, 255, 0), -1)
```

---

### Check calibration progress

```python
frame, vitals = sdk.run()

buf_len   = sdk.signal_extractor.length   # frames buffered so far
min_frames = sdk.MIN_FRAMES               # frames needed before first estimate (64)
progress  = min(100, buf_len / min_frames * 100)
print(f"Calibration: {progress:.0f}%")
```

---

### Reset the pipeline (e.g. when switching subjects)

```python
sdk.signal_extractor.reset()
sdk.heart_rate.reset()
sdk.spo2_estimator.reset()
sdk.bp_estimator.reset()
sdk.resp_estimator.reset()
```

---

## SDK API Reference

### `RPPGSDK`

Main class in `sdk/rppg_sdk.py`. Orchestrates the full pipeline.

| Method / Property | Description |
|---|---|
| `RPPGSDK(source, buffer_size, window_size)` | Constructor |
| `run()` | Process one frame. Returns `(frame, vitals)` |
| `release()` | Release camera and MediaPipe resources |
| `last_landmarks` | List of 468 `(x, y)` landmark coords, or `None` |
| `signal_extractor.length` | Current number of buffered frames |
| `MIN_FRAMES` | Minimum frames before estimates start (64) |
| `MIN_FRAMES_RESP` | Minimum frames for respiration (128) |

#### `vitals` dictionary keys

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `bpm` | `float \| None` | BPM | Heart rate |
| `spo2` | `float \| None` | % | Blood oxygen saturation |
| `bp_sys` | `float \| None` | mmHg | Systolic blood pressure |
| `bp_dia` | `float \| None` | mmHg | Diastolic blood pressure |
| `resp_rate` | `float \| None` | breaths/min | Respiration rate |

---

## Project Structure

```
rPPG-SDK/
├── algorithms/
│   └── pos.py                # POS rPPG algorithm (Wang et al., 2017)
├── sdk/
│   ├── camera.py             # Webcam capture interface
│   ├── face_detector.py      # MediaPipe face landmark detection
│   ├── roi_extractor.py      # Facial ROI segmentation
│   ├── signal_extractor.py   # Temporal RGB signal buffer
│   ├── rppg_algorithm.py     # Algorithm wrapper
│   ├── heart_rate.py         # Heart rate estimator
│   ├── vitals.py             # SpO2, BP, respiration estimators
│   └── rppg_sdk.py           # Main SDK orchestrator
├── examples/
│   └── webcam_demo.py        # Real-time webcam demo with dashboard UI
├── models/
│   └── face_landmarker.task  # MediaPipe face model (auto-downloaded on first run)
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Face Detection** — MediaPipe FaceLandmarker detects 468 facial landmarks per frame
2. **ROI Extraction** — Forehead and cheek regions are segmented using landmark-based convex hulls
3. **Signal Extraction** — Mean RGB values from ROIs are buffered with EMA smoothing into a temporal signal
4. **Pulse Extraction** — The POS algorithm isolates the blood volume pulse from the RGB fluctuations
5. **Vital Estimation** — Heart rate, blood pressure, SpO2, and respiration rate are each computed from the pulse and RGB signals

---

## Algorithm Details

### POS (Plane-Orthogonal-to-Skin)

Based on Wang et al., *"Algorithmic Principles of Remote PPG"* (IEEE TBME, 2017). Projects temporally-normalized RGB signals onto a plane orthogonal to skin tone variation, isolating the blood volume pulse.

### Heart Rate

- **Spectral**: Welch's PSD with 2048-point FFT and parabolic interpolation for sub-bin precision
- **Temporal**: Peak detection with adaptive prominence; median IBI → BPM
- **Fusion**: Weighted average when both agree within 15 BPM; history-closest used when they disagree
- **Smoothing**: Median filter over 5 recent estimates, ±20 BPM outlier gate

### SpO2

Ratio-of-ratios using AC/DC components of red and green channels as a proxy for oxygenated/deoxygenated hemoglobin absorption, empirically calibrated for webcam sensors.

### Blood Pressure

Pulse wave rising-time analysis combined with heart-rate regression modeling to estimate systolic and diastolic pressures.

### Respiration Rate

Hilbert transform extracts the amplitude envelope of the pulse (modulated by breathing). Welch spectral analysis on the envelope identifies the dominant respiratory frequency.

---

## Accuracy & Limitations

- Measurements require you to **sit still** with your face clearly lit and visible
- **Calibration** takes ~2–4 seconds (64–128 frames at 30 fps)
- Heart rate is most accurate after 5+ seconds of stable readings
- SpO2 and blood pressure are **estimates** — not medical-grade measurements
- Good, even, diffuse lighting significantly improves all readings

## Tips for Best Results

1. **Lighting** — Bright, even light on your face. Avoid backlighting or flickering sources
2. **Stability** — Keep your head still during a reading
3. **Distance** — Position your face 40–80 cm from the camera
4. **Glasses** — If wearing glasses, ensure the forehead and cheeks are still visible

---

## License

MIT

