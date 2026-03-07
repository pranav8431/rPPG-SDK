import os
import time
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model():
    """Download the face landmarker model if not already present."""
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"Downloading face landmarker model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Download complete.")


class FaceDetector:
    """Face landmark detector using MediaPipe FaceLandmarker (Tasks API)."""

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        """Initialize the FaceLandmarker.

        Args:
            max_num_faces: Maximum number of faces to detect.
            min_detection_confidence: Minimum confidence for detection.
        """
        _ensure_model()
        base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.monotonic()

    def detect(self, frame):
        """Detect facial landmarks in a BGR frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of (x, y) landmark pixel coordinates, or None if no face found.
        """
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = result.face_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        return points

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
