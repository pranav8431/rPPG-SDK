import numpy as np
from scipy.signal import detrend, butter, filtfilt


def _bandpass(signal, fps, lo=0.75, hi=3.0, order=4):
    """Apply a Butterworth bandpass filter (~45-180 BPM)."""
    nyq = fps / 2.0
    if hi >= nyq:
        hi = nyq - 0.1
    if lo >= hi:
        return signal
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def pos(signal, fps, window_size=32):
    """Plane-Orthogonal-to-Skin (POS) rPPG algorithm.

    Reference: Wang et al., "Algorithmic Principles of Remote PPG" (2017).

    Args:
        signal: Numpy array of shape (N, 3) with temporal RGB means.
        fps: Frame rate of the video source.
        window_size: Sliding window length in frames.

    Returns:
        1-D numpy array of the extracted pulse signal.
    """
    n = signal.shape[0]
    pulse = np.zeros(n)

    for t in range(window_size - 1, n):
        start = t - window_size + 1
        window = signal[start : t + 1, :]  # (window_size, 3)

        # Temporal normalization: divide each channel by its mean
        means = np.mean(window, axis=0)
        means[means == 0] = 1.0
        cn = window / means

        # POS projection matrix P = [[0, 1, -1], [-2, 1, 1]]
        xs = cn[:, 1] - cn[:, 2]              # G - B
        ys = -2.0 * cn[:, 0] + cn[:, 1] + cn[:, 2]  # -2R + G + B

        # Adaptive alpha
        std_xs = np.std(xs)
        std_ys = np.std(ys)
        alpha = std_xs / std_ys if std_ys > 1e-10 else 0.0

        # Pulse signal for this window
        p = xs + alpha * ys

        # Overlap-add
        pulse[start : t + 1] += (p - np.mean(p))

    # Detrend
    pulse = detrend(pulse)

    # Bandpass filter to isolate heart-rate frequencies
    if n >= 2 * window_size:
        pulse = _bandpass(pulse, fps)

    # Normalize
    std = np.std(pulse)
    if std > 1e-10:
        pulse = pulse / std

    return pulse
