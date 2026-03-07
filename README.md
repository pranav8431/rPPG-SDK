# rPPG SDK — Remote Photoplethysmography Vital Signs Monitor

A real-time, non-contact vital signs monitoring SDK that extracts physiological measurements from a standard webcam using remote photoplethysmography (rPPG).

## Features

- **Heart Rate (BPM)** — Dual spectral + time-domain estimation with parabolic interpolation for sub-BPM accuracy
- **Blood Pressure (mmHg)** — Systolic/diastolic estimation via pulse wave morphology analysis
- **Blood Oxygen Saturation (SpO2 %)** — Ratio-of-ratios technique on RGB channel pulsatile components
- **Respiration Rate (breaths/min)** — Hilbert transform envelope extraction with spectral peak analysis
- **Face Detection & Tracking** — MediaPipe FaceLandmarker with 468 landmarks for precise ROI extraction
- **Motion Detection** — Frame-to-frame motion indicator for signal quality awareness
- **Real-time Dashboard** — Live webcam feed with face overlay and vital signs display

## How It Works

1. **Face Detection** — MediaPipe FaceLandmarker detects 468 facial landmarks per frame
2. **ROI Extraction** — Forehead and cheek regions are segmented using landmark-based convex hulls
3. **Signal Extraction** — Mean RGB values from ROIs are buffered with EMA smoothing into a temporal signal
4. **Pulse Extraction** — The POS (Plane-Orthogonal-to-Skin) algorithm isolates the blood volume pulse from RGB fluctuations
5. **Vital Estimation** — Heart rate, blood pressure, SpO2, and respiration rate are computed from the pulse and RGB signals

## Project Structure

```
rppg-sdk/
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
│   └── face_landmarker.task  # MediaPipe face model (auto-downloaded)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/pranav8431/rPPG-SDK
cd rppg-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- OpenCV >= 4.5
- MediaPipe >= 0.10
- NumPy >= 1.21
- SciPy >= 1.7

## Quick Start

### Run the Demo

```bash
python examples/webcam_demo.py
```

Press **q** to quit. The demo shows a live webcam feed with:
- Face contour overlay
- Calibration progress circle (fills during initial ~2 second buffering)
- Motion indicator
- Bottom dashboard with all vital signs

### Use the SDK Programmatically

```python
from sdk.rppg_sdk import RPPGSDK

sdk = RPPGSDK(source=0)  # 0 = default webcam

while True:
    frame, vitals = sdk.run()
    if frame is None:
        break

    bpm       = vitals['bpm']        # Heart rate in BPM
    spo2      = vitals['spo2']       # Blood oxygen %
    bp_sys    = vitals['bp_sys']     # Systolic blood pressure (mmHg)
    bp_dia    = vitals['bp_dia']     # Diastolic blood pressure (mmHg)
    resp_rate = vitals['resp_rate']  # Respiration rate (breaths/min)

    if bpm is not None:
        print(f"HR: {bpm:.0f} bpm | SpO2: {spo2:.0f}% | BP: {bp_sys:.0f}/{bp_dia:.0f} | RR: {resp_rate:.0f}")

sdk.release()
```

### Configuration

```python
sdk = RPPGSDK(
    source=0,           # Camera index or video file path
    buffer_size=256,     # Signal buffer length (frames)
    window_size=32,      # POS algorithm window size
)
```

## Algorithm Details

### POS (Plane-Orthogonal-to-Skin)

Based on Wang et al., *"Algorithmic Principles of Remote PPG"* (IEEE TBME, 2017). Projects temporally-normalized RGB signals onto a plane orthogonal to skin tone variation, isolating the blood volume pulse.

### Heart Rate Estimation

- **Spectral**: Welch's PSD with 2048-point FFT and parabolic interpolation for sub-bin frequency accuracy
- **Temporal**: Peak detection with adaptive prominence thresholding
- **Fusion**: Weighted average when both methods agree; history-based selection when they diverge
- **Smoothing**: Median filter over recent estimates with ±20 BPM outlier rejection

### SpO2 Estimation

Ratio-of-ratios method using AC/DC components of red and green channels as proxy for oxygenated/deoxygenated hemoglobin, with empirical webcam calibration.

### Blood Pressure Estimation

Pulse wave analysis measuring rising time ratios and cardiac period, combined with heart-rate regression modeling.

### Respiration Rate

Hilbert transform extracts the amplitude envelope of the pulse signal, which is modulated by breathing. Welch spectral analysis on the envelope identifies the respiratory frequency.

## Accuracy Notes

- Measurements require the user to **sit still** with face clearly visible
- **Calibration** takes approximately 2-4 seconds (64-128 frames)
- Heart rate accuracy is highest after 5+ seconds of stable readings
- SpO2 and blood pressure are **estimates** — not medical-grade measurements
- Respiration rate requires ~4 seconds of data for reliable estimation
- Good, even lighting significantly improves signal quality

## Tips for Best Results

1. **Lighting** — Use consistent, diffuse lighting. Avoid strong shadows or flickering lights
2. **Stability** — Keep your head relatively still during measurement
3. **Distance** — Position face 40-80 cm from the camera
4. **Camera** — Higher resolution and frame rate improve accuracy

## License

MIT
