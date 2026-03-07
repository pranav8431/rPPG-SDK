import sys
import os
import numpy as np

# Add project root to path so algorithms package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algorithms.pos import pos


class RPPGAlgorithm:
    """Wrapper for rPPG pulse extraction algorithms."""

    def __init__(self, algorithm="pos", window_size=32):
        """Initialize the rPPG algorithm.

        Args:
            algorithm: Algorithm name. Currently only 'pos' is supported.
            window_size: Sliding window length in frames for the algorithm.
        """
        if algorithm != "pos":
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        self.algorithm = algorithm
        self.window_size = window_size

    def process(self, signal, fps):
        """Extract pulse signal from RGB temporal signal.

        Args:
            signal: Numpy array of shape (N, 3) with RGB values.
            fps: Frame rate of the source video.

        Returns:
            1-D pulse signal array, or None if signal is too short.
        """
        if signal is None or len(signal) < self.window_size:
            return None
        return pos(signal, fps, window_size=self.window_size)
