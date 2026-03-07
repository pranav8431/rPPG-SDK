import cv2


class Camera:
    """Webcam interface using OpenCV VideoCapture."""

    def __init__(self, source=0):
        """Initialize camera.

        Args:
            source: Camera device index or video file path.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

    @property
    def fps(self):
        """Return the camera frame rate."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def read(self):
        """Read a single frame from the camera.

        Returns:
            Frame as a numpy array (BGR), or None if read fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the camera resource."""
        self.cap.release()
