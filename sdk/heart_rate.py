import numpy as np
from scipy.signal import welch, find_peaks
from collections import deque


class HeartRate:
    """Estimates heart rate (BPM) from a pulse signal using combined spectral
    and time-domain analysis with temporal smoothing."""

    MIN_BPM = 45.0
    MAX_BPM = 180.0

    def __init__(self, smooth_window=5):
        self._history = deque(maxlen=smooth_window)

    def compute(self, pulse_signal, fps):
        """Compute heart rate from a pulse signal.

        Uses Welch's PSD with parabolic interpolation for sub-bin precision,
        cross-validated against time-domain peak detection.

        Args:
            pulse_signal: 1-D numpy array of the pulse waveform.
            fps: Sampling rate (frames per second).

        Returns:
            Smoothed BPM as a float, or None if estimation fails.
        """
        if pulse_signal is None or len(pulse_signal) < 64:
            return None

        spectral_bpm = self._spectral_estimate(pulse_signal, fps)
        temporal_bpm = self._temporal_estimate(pulse_signal, fps)

        # Combine estimates: prefer spectral, use temporal as validation
        if spectral_bpm is None and temporal_bpm is None:
            return None

        if spectral_bpm is not None and temporal_bpm is not None:
            # If both agree within 15 BPM, average them (spectral weighted more)
            if abs(spectral_bpm - temporal_bpm) < 15:
                raw_bpm = 0.7 * spectral_bpm + 0.3 * temporal_bpm
            else:
                # Disagree — trust whichever is closer to recent history
                raw_bpm = self._pick_closest(spectral_bpm, temporal_bpm)
        elif spectral_bpm is not None:
            raw_bpm = spectral_bpm
        else:
            raw_bpm = temporal_bpm

        # Outlier rejection
        if len(self._history) >= 3:
            median = np.median(self._history)
            if abs(raw_bpm - median) > 20:
                return self._smoothed_bpm()

        self._history.append(raw_bpm)
        return self._smoothed_bpm()

    def _spectral_estimate(self, pulse, fps):
        """Welch PSD with parabolic interpolation for sub-bin accuracy."""
        nperseg = min(len(pulse), 256)
        nfft = 2048  # high zero-padding for fine freq resolution
        freqs, psd = welch(pulse, fs=fps, nperseg=nperseg,
                           noverlap=nperseg * 3 // 4, nfft=nfft)

        min_freq = self.MIN_BPM / 60.0
        max_freq = self.MAX_BPM / 60.0
        mask = (freqs >= min_freq) & (freqs <= max_freq)

        if not np.any(mask):
            return None

        valid_freqs = freqs[mask]
        valid_psd = psd[mask]

        peak_idx = np.argmax(valid_psd)

        # Parabolic interpolation around the peak for sub-bin precision
        if 0 < peak_idx < len(valid_psd) - 1:
            alpha = valid_psd[peak_idx - 1]
            beta = valid_psd[peak_idx]
            gamma = valid_psd[peak_idx + 1]
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-12:
                correction = 0.5 * (alpha - gamma) / denom
                df = valid_freqs[1] - valid_freqs[0] if len(valid_freqs) > 1 else 0
                peak_freq = valid_freqs[peak_idx] + correction * df
            else:
                peak_freq = valid_freqs[peak_idx]
        else:
            peak_freq = valid_freqs[peak_idx]

        bpm = peak_freq * 60.0
        if self.MIN_BPM <= bpm <= self.MAX_BPM:
            return bpm
        return None

    def _temporal_estimate(self, pulse, fps):
        """Time-domain peak counting for BPM estimation."""
        min_dist = max(int(fps * 60.0 / self.MAX_BPM), 1)
        prominence = max(np.std(pulse) * 0.4, 0.05)
        peaks, _ = find_peaks(pulse, distance=min_dist, prominence=prominence)

        if len(peaks) < 3:
            return None

        intervals = np.diff(peaks) / fps  # seconds between beats
        # Filter out physiologically impossible intervals
        valid = (intervals > 60.0 / self.MAX_BPM) & (intervals < 60.0 / self.MIN_BPM)
        intervals = intervals[valid]

        if len(intervals) < 2:
            return None

        # Use median interval for robustness against occasional missed beats
        median_interval = float(np.median(intervals))
        bpm = 60.0 / median_interval
        if self.MIN_BPM <= bpm <= self.MAX_BPM:
            return bpm
        return None

    def _pick_closest(self, bpm1, bpm2):
        """Pick whichever BPM is closer to recent history."""
        if not self._history:
            return bpm1
        median = np.median(self._history)
        return bpm1 if abs(bpm1 - median) <= abs(bpm2 - median) else bpm2

    def _smoothed_bpm(self):
        if len(self._history) == 0:
            return None
        return float(np.median(self._history))

    def reset(self):
        self._history.clear()
