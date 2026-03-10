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

        Uses three independent methods (spectral, temporal peak detection,
        and autocorrelation) with majority-vote fusion for robust estimation.

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
        autocorr_bpm = self._autocorr_estimate(pulse_signal, fps)

        estimates = [e for e in [spectral_bpm, temporal_bpm, autocorr_bpm]
                     if e is not None]

        if len(estimates) == 0:
            return None

        if len(estimates) == 1:
            raw_bpm = estimates[0]
        elif len(estimates) == 2:
            if abs(estimates[0] - estimates[1]) < 12:
                raw_bpm = np.mean(estimates)
            else:
                raw_bpm = self._pick_closest(estimates[0], estimates[1])
        else:
            # Three estimates: find the closest agreeing pair
            est = sorted(estimates)
            pairs = [(est[0], est[1]), (est[1], est[2]), (est[0], est[2])]
            diffs = [abs(a - b) for a, b in pairs]
            best_idx = int(np.argmin(diffs))
            a, b = pairs[best_idx]
            if diffs[best_idx] < 12:
                raw_bpm = (a + b) / 2.0
            else:
                raw_bpm = est[1]  # median of three

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

        # Harmonic rejection: if peak could be a 2nd harmonic,
        # check for significant energy at the sub-harmonic frequency
        half_freq = peak_freq / 2.0
        if half_freq >= min_freq:
            half_idx = np.argmin(np.abs(valid_freqs - half_freq))
            if valid_psd[half_idx] > 0.35 * valid_psd[peak_idx]:
                peak_freq = valid_freqs[half_idx]

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

        # IQR-based outlier removal for cleaner interval estimation
        if len(intervals) >= 4:
            q1, q3 = np.percentile(intervals, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                iqr_mask = ((intervals >= q1 - 1.5 * iqr)
                            & (intervals <= q3 + 1.5 * iqr))
                filtered = intervals[iqr_mask]
                if len(filtered) >= 2:
                    intervals = filtered

        # Use median interval for robustness against occasional missed beats
        median_interval = float(np.median(intervals))
        bpm = 60.0 / median_interval
        if self.MIN_BPM <= bpm <= self.MAX_BPM:
            return bpm
        return None

    def _autocorr_estimate(self, pulse, fps):
        """Autocorrelation-based BPM — robust to harmonics."""
        n = len(pulse)
        if n < 64:
            return None
        pulse_c = pulse - np.mean(pulse)
        acf = np.correlate(pulse_c, pulse_c, mode='full')[n - 1:]
        if acf[0] < 1e-10:
            return None
        acf /= acf[0]

        min_lag = max(int(fps * 60.0 / self.MAX_BPM), 1)
        max_lag = min(int(fps * 60.0 / self.MIN_BPM), n - 1)
        if min_lag >= max_lag:
            return None

        search = acf[min_lag:max_lag + 1]
        peaks, _ = find_peaks(search, prominence=0.1, height=0.2)
        if len(peaks) == 0:
            return None

        best = peaks[np.argmax(search[peaks])]
        lag = best + min_lag
        bpm = 60.0 * fps / lag
        return bpm if self.MIN_BPM <= bpm <= self.MAX_BPM else None

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
