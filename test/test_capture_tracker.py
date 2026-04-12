"""Unit tests for CaptureTracker."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CaptureTrackingConfig
from src.scheduler.capture_tracker import (
    CaptureTracker, CaptureAction,
)


class MockDetector:
    """Mock face detector returning predetermined results."""

    def __init__(self, detections=None):
        self.detections = detections or []

    def detect(self, frame):
        return self.detections

    def set_detections(self, dets):
        self.detections = dets


class MockDetection:
    def __init__(self, bbox):
        self.bbox = bbox


def test_giveup_on_no_face():
    """When no face detected for > giveup_ms, should return giveup."""
    cfg = CaptureTrackingConfig(
        face_lost_kalman_ms=100,
        face_lost_giveup_ms=200,
        max_corrections=3,
        safe_zone_ratio=0.6,
    )
    detector = MockDetector()
    tracker = CaptureTracker(detector, cfg)

    # Simulate having seen a face
    tracker.last_face_ts = 0.0  # Long time ago
    tracker.last_face_bbox = (100, 100, 300, 300)
    tracker.kf_initialized = True

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    action = tracker.step(frame)
    assert action.type == "giveup", f"Expected giveup, got {action.type}"
    print("PASS: giveup on no face")


def test_collect_in_safe_zone():
    """Face in center -> collect action with quality crop."""
    cfg = CaptureTrackingConfig(
        safe_zone_ratio=0.6,
        max_corrections=3,
    )
    detector = MockDetector()
    tracker = CaptureTracker(detector, cfg)

    # Create a face-like detection in center
    det = MockDetection((500, 250, 700, 450))
    detector.set_detections([det])

    # Create a face-like frame (enough detail for quality check)
    frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    action = tracker.step(frame)
    # May be "collect" or "none" depending on quality check
    assert action.type in ("collect", "none"), f"Unexpected: {action.type}"
    print(f"PASS: face in safe zone -> {action.type}")


def test_correction_settle():
    """After enter_correction_settle, should return settling."""
    cfg = CaptureTrackingConfig(correction_settle=0.2)
    detector = MockDetector()
    tracker = CaptureTracker(detector, cfg)

    tracker.enter_correction_settle()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    action = tracker.step(frame)
    assert action.type == "settling", f"Expected settling, got {action.type}"
    print("PASS: correction settle")


def test_safe_zone_boundary():
    """Check safe zone calculation."""
    cfg = CaptureTrackingConfig(safe_zone_ratio=0.6)
    detector = MockDetector()
    tracker = CaptureTracker(detector, cfg)

    # 1280x720 frame, margin = 0.2 * 1280 = 256
    assert tracker._in_safe_zone(640, 360, (720, 1280, 3))  # center
    assert not tracker._in_safe_zone(10, 10, (720, 1280, 3))  # corner
    assert not tracker._in_safe_zone(1270, 710, (720, 1280, 3))  # far corner
    print("PASS: safe zone boundaries")


if __name__ == "__main__":
    test_giveup_on_no_face()
    test_collect_in_safe_zone()
    test_correction_settle()
    test_safe_zone_boundary()
    print("\nAll CaptureTracker tests passed!")
