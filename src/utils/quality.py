"""Face quality checks for capture filtering."""

import cv2
import numpy as np


def quality_ok(face_bgr: np.ndarray,
               min_w: int = 80,
               ar_lo: float = 0.6,
               ar_hi: float = 1.6,
               blur_th: float = 50.0) -> bool:
    """Check if a face crop meets quality thresholds.

    Checks:
    - Width >= min_w pixels
    - Aspect ratio (h/w) within [ar_lo, ar_hi]
    - Laplacian variance >= blur_th (sharpness)
    """
    h, w = face_bgr.shape[:2]
    if w < min_w:
        return False
    ar = h / (w + 1e-6)
    if ar < ar_lo or ar > ar_hi:
        return False
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var >= blur_th
