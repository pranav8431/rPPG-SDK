import numpy as np
from scipy.signal import find_peaks, welch, butter, filtfilt, hilbert
from collections import deque


def _bandpass(signal, fps, lo, hi, order=3):
    """Apply Butterworth bandpass filter."""
    nyq = fps / 2.0
    if hi >= nyq:
        hi = nyq - 0.1
    if lo >= hi:
        return signal
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


class SpO2Estimator:
    """Blood oxygen saturation using ratio of ratios from RGB channels."""

    def __init__(self, smooth_window=11):
        self._history = deque(maxlen=smooth_window)

    def compute(self, rgb_signal, fps):
        """Estimate SpO2 from temporal RGB signal.

        Uses red and green channels as proxy for oxygenated/deoxygenated
        hemoglobin absorption in webcam RGB data.

        Args:
            rgb_signal: Array of shape (N, 3) with RGB means.
            fps: Sampling rate.
        """
        if rgb_signal is None or len(rgb_signal) < 64:
            return None

        red = rgb_signal[:, 0].astype(np.float64)
        green = rgb_signal[:, 1].astype(np.float64)

        dc_red = np.mean(red)
        dc_green = np.mean(green)

        if dc_red < 1.0 or dc_green < 1.0:
            return None

        # Extract pulsatile (AC) components via bandpass filtering
        red_ac = _bandpass(red, fps, 0.7, 3.5)
        green_ac = _bandpass(green, fps, 0.7, 3.5)

        ac_red = np.std(red_ac)
        ac_green = np.std(green_ac)

        if ac_green < 1e-10:
            return None

        # Ratio of Ratios (red/green channels as proxy)
        ror = (ac_red / dc_red) / (ac_green / dc_green)

        # Webcam-calibrated formula: healthy subjects typically yield
        # RoR ~ 0.4-0.8 mapping to SpO2 95-99%
        spo2 = 100.0 - 5.0 * ror
        spo2 = float(np.clip(spo2, 92, 100))

        self._history.append(spo2)
        return self._smoothed()

    def _smoothed(self):
        if not self._history:
            return None
        return float(np.median(self._history))

    def reset(self):
        self._history.clear()


class RespirationEstimator:
    """Respiration rate from respiratory-induced pulse amplitude modulation."""

    def __init__(self, smooth_window=5):
        self._history = deque(maxlen=smooth_window)

    def compute(self, pulse_signal, fps):
        """Extract respiration rate from pulse signal envelope."""
        if pulse_signal is None or len(pulse_signal) < 128:
            return None

        # Amplitude envelope via analytic signal (Hilbert transform)
        analytic = hilbert(pulse_signal)
        envelope = np.abs(analytic)

        # Bandpass for respiratory frequencies (6-24 breaths/min)
        resp_signal = _bandpass(envelope, fps, 0.1, 0.4)

        if np.std(resp_signal) < 1e-10:
            return None

        # Spectral peak detection
        nperseg = min(len(resp_signal), 256)
        freqs, psd = welch(resp_signal, fs=fps, nperseg=nperseg, nfft=1024)

        mask = (freqs >= 0.1) & (freqs <= 0.4)
        if not np.any(mask):
            return None

        valid_freqs = freqs[mask]
        valid_psd = psd[mask]

        peak_idx = np.argmax(valid_psd)
        rr = valid_freqs[peak_idx] * 60.0

        if 6 < rr < 25:
            self._history.append(rr)

        return self._smoothed()

    def _smoothed(self):
        if not self._history:
            return None
        return float(np.median(self._history))

    def reset(self):
        self._history.clear()


class BloodPressureEstimator:
    """Blood pressure from multi-feature PPG pulse waveform morphology.

    Two morphological features are extracted per complete cardiac beat:

    rise_ratio = foot-to-systolic-peak time / cycle length
        Proxy for arterial stiffness. Stiffer arteries propagate the pressure
        wave faster → shorter upstroke → lower rise_ratio → higher SBP.
        Healthy resting adults: ~0.30–0.35.

    fall_ratio = systolic-peak to next-foot time / cycle length
        Proxy for peripheral vascular resistance. Higher resistance slows
        diastolic run-off → longer fall phase → higher fall_ratio → higher DBP.
        Healthy resting adults: ~0.65–0.70.

    IBI (inter-beat interval) is derived directly from the waveform and used
    in place of the passed heart_rate for the HR-deviation term when the
    signal is clean enough to yield valid beats.
    """

    def __init__(self, smooth_window=11):
        self._sys_history = deque(maxlen=smooth_window)
        self._dia_history = deque(maxlen=smooth_window)

    def compute(self, pulse_signal, fps, heart_rate):
        """Estimate systolic and diastolic blood pressure.

        Returns:
            Tuple of (systolic, diastolic) in mmHg, or (None, None).
        """
        if pulse_signal is None or len(pulse_signal) < 64 or heart_rate is None:
            return self._smoothed_sys(), self._smoothed_dia()

        # Bandpass to cardiac band for clean morphology, then unit-normalise
        sig = _bandpass(pulse_signal.copy(), fps, 0.7, 3.0, order=3)
        sig -= np.mean(sig)
        std = np.std(sig)
        if std < 1e-10:
            return self._smoothed_sys(), self._smoothed_dia()
        sig /= std

        features = self._extract_beat_features(sig, fps)
        if not features or len(features) < 2:
            return self._smoothed_sys(), self._smoothed_dia()

        rise_ratio = float(np.median([f[0] for f in features]))
        fall_ratio = float(np.median([f[1] for f in features]))
        ibi        = float(np.median([f[2] for f in features]))  # seconds
        pw50       = float(np.median([f[3] for f in features]))

        # HR computed from IBI is more accurate than the passed value when
        # the waveform yields enough clean beats.
        hr_from_ibi = 60.0 / ibi if 0.33 < ibi < 2.0 else heart_rate
        hr_dev = hr_from_ibi - 72.0   # deviation from normal resting HR

        # Stiffness term: healthy rise_ratio ≈ 0.32
        # Lower rise_ratio → faster upstroke → stiffer vessels → higher SBP
        stiffness  = 0.32 - rise_ratio   # positive = stiffer than normal

        # Resistance term: healthy fall_ratio ≈ 0.68
        # Higher fall_ratio → slower run-off → higher peripheral resistance → higher DBP
        resistance = fall_ratio - 0.68   # positive = higher resistance than normal

        # Pulse width term: healthy pw50 ≈ 0.42
        # Wider pulse at half-height → lower vascular resistance → lower BP
        width_dev = pw50 - 0.42

        sys = 120.0 + hr_dev * 0.50 + stiffness * 60.0 - width_dev * 15.0
        dia =  80.0 + hr_dev * 0.25 + resistance * 35.0 - width_dev * 25.0

        sys = float(np.clip(sys, 95, 160))
        dia = float(np.clip(dia, 55, 100))

        # Enforce minimum physiological pulse pressure (>= 20 mmHg)
        if sys - dia < 20:
            centre = (sys + dia) / 2.0
            sys = centre + 10.0
            dia = centre - 10.0

        self._sys_history.append(sys)
        self._dia_history.append(dia)
        return self._smoothed_sys(), self._smoothed_dia()

    def _extract_beat_features(self, sig, fps):
        """Extract (rise_ratio, fall_ratio, IBI) for each complete beat.

        The signal must already be bandpassed and unit-normalised before
        calling this method.
        """
        min_dist = max(int(fps * 0.33), 1)   # at most ~180 BPM between events

        peaks,   _ = find_peaks( sig, distance=min_dist, prominence=0.30)
        troughs, _ = find_peaks(-sig, distance=min_dist, prominence=0.20)

        if len(peaks) < 3 or len(troughs) < 3:
            return None

        feats = []
        for pk in peaks:
            pre  = troughs[troughs < pk]
            post = troughs[troughs > pk]
            if len(pre) == 0 or len(post) == 0:
                continue

            t0 = int(pre[-1])   # foot before peak
            t2 = int(post[0])   # foot after peak
            t1 = int(pk)        # systolic peak

            cycle = t2 - t0
            # Reject cycles outside physiological HR range (30–180 BPM)
            if not (fps * 0.33 < cycle < fps * 2.0):
                continue

            # Amplitude above interpolated baseline — rejects noise spikes
            amp = sig[t1] - 0.5 * (sig[t0] + sig[t2])
            if amp < 0.25:
                continue

            rise = (t1 - t0) / cycle
            fall = (t2 - t1) / cycle

            # PW50: pulse width at 50% of beat amplitude
            baseline = 0.5 * (sig[t0] + sig[t2])
            half_height = baseline + 0.5 * (sig[t1] - baseline)
            above_half = sig[t0:t2 + 1] > half_height
            pw50 = float(np.sum(above_half)) / cycle

            # Morphological plausibility bounds
            if 0.10 <= rise <= 0.55 and 0.45 <= fall <= 0.90:
                feats.append((rise, fall, cycle / fps, pw50))

        return feats if len(feats) >= 2 else None

    def _smoothed_sys(self):
        return float(np.median(self._sys_history)) if self._sys_history else None

    def _smoothed_dia(self):
        return float(np.median(self._dia_history)) if self._dia_history else None

    def reset(self):
        self._sys_history.clear()
        self._dia_history.clear()
