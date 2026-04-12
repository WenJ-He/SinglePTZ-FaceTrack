"""Geometry utilities for bbox operations."""

from typing import Tuple
import numpy as np


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Return (cx, cy) center of bbox."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    """Return area of bbox."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_expand(bbox: Tuple[int, int, int, int], ratio: float,
                frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Expand bbox by ratio around center, clipped to frame boundaries."""
    cx, cy = bbox_center(bbox)
    w = (bbox[2] - bbox[0]) * ratio / 2
    h = (bbox[3] - bbox[1]) * ratio / 2
    x1 = max(int(cx - w), 0)
    y1 = max(int(cy - h), 0)
    x2 = min(int(cx + w), frame_w)
    y2 = min(int(cy + h), frame_h)
    return (x1, y1, x2, y2)


def bbox_to_point_frame(bbox: Tuple[int, int, int, int],
                        frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Convert pixel bbox to normalized [0, 255] coordinates for SDK POINT_FRAME."""
    return (
        int(bbox[0] * 255 / frame_w),
        int(bbox[1] * 255 / frame_h),
        int(bbox[2] * 255 / frame_w),
        int(bbox[3] * 255 / frame_h),
    )


def iou(a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int]) -> float:
    """Intersection over Union of two bboxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
