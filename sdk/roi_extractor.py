import cv2
import numpy as np


# MediaPipe FaceMesh landmark indices — verified forehead-only region
_FOREHEAD_INDICES = [10, 67, 69, 104, 108, 109, 151, 299, 297, 337, 338, 333, 332, 334]
_LEFT_CHEEK_INDICES = [36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 100, 126]
_RIGHT_CHEEK_INDICES = [266, 425, 426, 427, 411, 352, 345, 346, 347, 348, 329, 355]


class ROIExtractor:
    """Extracts region-of-interest pixel values from facial landmarks."""

    def __init__(self, regions=None):
        """Initialize ROI extractor.

        Args:
            regions: List of region names to use. Options: 'forehead', 'left_cheek', 'right_cheek'.
                     Defaults to all three.
        """
        self.regions = regions or ["forehead", "left_cheek", "right_cheek"]
        self._region_map = {
            "forehead": _FOREHEAD_INDICES,
            "left_cheek": _LEFT_CHEEK_INDICES,
            "right_cheek": _RIGHT_CHEEK_INDICES,
        }

    def extract(self, frame, landmarks):
        """Extract mean RGB values from ROI regions.

        Args:
            frame: BGR image as numpy array.
            landmarks: List of (x, y) landmark pixel coordinates.

        Returns:
            Mean RGB values as numpy array of shape (3,), or None if extraction fails.
        """
        all_means = []
        h, w = frame.shape[:2]

        for region_name in self.regions:
            indices = self._region_map[region_name]
            pts = np.array(
                [landmarks[i] for i in indices if i < len(landmarks)],
                dtype=np.int32,
            )
            if len(pts) < 3:
                continue

            # Build a mask for the convex hull of the ROI
            mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

            # Extract pixels inside the mask
            roi_pixels = frame[mask == 255]
            if len(roi_pixels) == 0:
                continue

            # Per-region mean (more robust than pooling all pixels)
            all_means.append(np.mean(roi_pixels, axis=0))

        if not all_means:
            return None

        # Average across regions, convert BGR→RGB
        mean_bgr = np.mean(all_means, axis=0)
        mean_rgb = mean_bgr[::-1].copy()
        return mean_rgb
