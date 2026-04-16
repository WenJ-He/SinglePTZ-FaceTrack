"""Sticky single-target tracker for early-stage flows."""

import time
from dataclasses import dataclass

from src.geometry import iou
from src.target_selector import SingleTargetSelector


@dataclass
class TrackedTarget:
    bbox: tuple[int, int, int, int]
    score: float
    last_seen_ts: float


class SingleTargetTracker:
    """Keep one sticky target using IoU matching with a timeout fallback."""

    def __init__(self, lost_timeout_s: float = 1.0, min_iou: float = 0.1):
        self.lost_timeout_s = lost_timeout_s
        self.min_iou = min_iou
        self.current: TrackedTarget | None = None

    def reset(self) -> None:
        self.current = None

    def update(self, detections, frame_shape, now: float | None = None) -> TrackedTarget | None:
        now = time.time() if now is None else now
        detections = list(detections)
        if not detections:
            if self.current and (now - self.current.last_seen_ts) <= self.lost_timeout_s:
                return self.current
            self.current = None
            return None

        chosen = None
        if self.current is not None:
            chosen = max(detections, key=lambda det: iou(det.bbox, self.current.bbox))
            if iou(chosen.bbox, self.current.bbox) < self.min_iou:
                chosen = None

        if chosen is None:
            chosen = SingleTargetSelector.pick_primary(detections, frame_shape)

        self.current = TrackedTarget(
            bbox=chosen.bbox,
            score=float(getattr(chosen, "score", 0.0)),
            last_seen_ts=now,
        )
        return self.current
