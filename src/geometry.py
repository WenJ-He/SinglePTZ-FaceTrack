"""Geometry helpers used by the refactored runtime."""

from typing import Tuple


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_bbox(x: float, y: float, bbox: Tuple[int, int, int, int]) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def is_edge_bbox(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    margin: int = 5,
) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 < margin or y1 < margin or x2 > frame_w - margin or y2 > frame_h - margin
