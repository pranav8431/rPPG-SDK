from sdk.camera import Camera
from sdk.face_detector import FaceDetector
from sdk.roi_extractor import ROIExtractor
from sdk.signal_extractor import SignalExtractor
from sdk.rppg_algorithm import RPPGAlgorithm
from sdk.heart_rate import HeartRate
from sdk.vitals import SpO2Estimator, RespirationEstimator, BloodPressureEstimator


class RPPGSDK:
    """Main rPPG SDK orchestrating the full heart-rate estimation pipeline."""

    # Minimum frames needed before attempting BPM estimation
    MIN_FRAMES = 64
    MIN_FRAMES_RESP = 128

    def __init__(self, source=0, buffer_size=256, window_size=32):
        """Initialize all pipeline modules.

        Args:
            source: Camera device index or video file path.
            buffer_size: Number of frames to keep in the signal buffer.
            window_size: Sliding window size for the POS algorithm.
        """
        self.camera = Camera(source)
        self.fps = self.camera.fps
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.signal_extractor = SignalExtractor(buffer_size=buffer_size)
        self.rppg_algorithm = RPPGAlgorithm(window_size=window_size)
        self.heart_rate = HeartRate(smooth_window=7)
        self.spo2_estimator = SpO2Estimator()
        self.resp_estimator = RespirationEstimator()
        self.bp_estimator = BloodPressureEstimator()
        self._no_face_count = 0
        self.last_landmarks = None

    def run(self):
        """Process one frame through the full pipeline.

        Returns:
            Tuple of (frame, vitals) where vitals is a dict with keys:
            'bpm', 'spo2', 'bp_sys', 'bp_dia', 'resp_rate'.
        """
        empty = {'bpm': None, 'spo2': None,
                 'bp_sys': None, 'bp_dia': None, 'resp_rate': None}

        frame = self.camera.read()
        if frame is None:
            return None, empty

        landmarks = self.face_detector.detect(frame)
        self.last_landmarks = landmarks

        if landmarks is None:
            self._no_face_count += 1
            if self._no_face_count > self.fps:
                self.signal_extractor.reset()
                self.heart_rate.reset()
                self.spo2_estimator.reset()
                self.resp_estimator.reset()
                self.bp_estimator.reset()
            return frame, empty
        self._no_face_count = 0

        rgb_mean = self.roi_extractor.extract(frame, landmarks)
        if rgb_mean is None:
            return frame, empty

        self.signal_extractor.update(rgb_mean)

        if self.signal_extractor.length < self.MIN_FRAMES:
            return frame, empty

        signal = self.signal_extractor.get_signal()
        pulse = self.rppg_algorithm.process(signal, self.fps)

        bpm = self.heart_rate.compute(pulse, self.fps)
        spo2 = self.spo2_estimator.compute(signal, self.fps)
        bp_sys, bp_dia = self.bp_estimator.compute(pulse, self.fps, bpm)

        resp_rate = None
        if self.signal_extractor.length >= self.MIN_FRAMES_RESP:
            resp_rate = self.resp_estimator.compute(pulse, self.fps)

        return frame, {'bpm': bpm, 'spo2': spo2,
                       'bp_sys': bp_sys, 'bp_dia': bp_dia,
                       'resp_rate': resp_rate}

    def release(self):
        """Release all resources."""
        self.camera.release()
        self.face_detector.close()
