import numpy as np
from collections import deque


class SignalExtractor:
    """Maintains a temporal buffer of RGB signal values with EMA smoothing."""

    def __init__(self, buffer_size=256, ema_alpha=0.4):
        """Initialize the signal buffer.

        Args:
            buffer_size: Maximum number of frames to keep in the buffer.
            ema_alpha: Smoothing factor for exponential moving average (0=max smooth, 1=no smooth).
        """
        self.buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)
        self._ema_alpha = ema_alpha
        self._prev = None

    def update(self, rgb_mean):
        """Add a new RGB mean value to the buffer with EMA smoothing.

        Args:
            rgb_mean: Array of shape (3,) with mean R, G, B values.
        """
        val = rgb_mean.copy()
        if self._prev is not None:
            val = self._ema_alpha * val + (1.0 - self._ema_alpha) * self._prev
        self._prev = val.copy()
        self._buffer.append(val)

    def get_signal(self):
        """Return the buffered RGB signal.

        Returns:
            Numpy array of shape (N, 3) or None if buffer is empty.
        """
        if len(self._buffer) == 0:
            return None
        return np.array(self._buffer)

    @property
    def length(self):
        """Current number of frames in the buffer."""
        return len(self._buffer)

    def reset(self):
        """Clear the signal buffer."""
        self._buffer.clear()
        self._prev = None
