"""CAPTURE stage dynamic tracking correction (Layer B+C strategy)."""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import CaptureTrackingConfig
from src.detect.yolo_face import YoloFace
from src.utils.quality import quality_ok
from src.sdk.hik_isapi import HikISAPI

logger = logging.getLogger("app")


@dataclass
class CaptureAction:
    type: str  # "collect" / "correct" / "settling" / "giveup" / "none"
    face_crop: Optional[np.ndarray] = None
    corrected_bbox: Optional[Tuple[int, int, int, int]] = None


class CaptureTracker:
    """CAPTURE stage tracking correction controller.

    Layer C: Safety margin via 2.0x expand in zoom_to_bbox
    Layer B: Discrete 3D re-positioning when face drifts to danger zone
    Layer B+: Kalman prediction for brief frame loss
    """

    def __init__(self, face_detector: YoloFace,
                 cfg: CaptureTrackingConfig,
                 isapi: Optional[HikISAPI] = None):
        self.detector = face_detector
        self.cfg = cfg
        self.isapi = isapi
        self.kf = self._init_kalman()
        self.kf_initialized = False

        self.last_face_ts: float = 0.0
        self.last_face_bbox: Optional[Tuple] = None
        self.correction_count: int = 0
        self.in_correction_settle: bool = False
        self.settle_until: float = 0.0
        self._prev_gray: Optional[np.ndarray] = None

    def step(self, frame: np.ndarray) -> CaptureAction:
        """Process one frame, return action decision."""
        now = time.time()

        # In correction settle
        if self.in_correction_settle:
            if now < self.settle_until:
                return CaptureAction("settling")
            self.in_correction_settle = False
            self._prev_gray = None

        # Face detection
        dets = self.detector.detect(frame)
        face = self._pick_center_largest(dets, frame.shape) if dets else None

        if face is not None:
            face_cx = (face.bbox[0] + face.bbox[2]) / 2
            face_cy = (face.bbox[1] + face.bbox[3]) / 2
            self._kalman_update(face_cx, face_cy)
            self.last_face_ts = now
            self.last_face_bbox = face.bbox

            if self._in_safe_zone(face_cx, face_cy, frame.shape):
                # Safe zone: collect
                # Prefer ISAPI high-quality frame if available
                source_frame = frame
                if self.isapi is not None:
                    isapi_frame = self.isapi.capture_jpeg()
                    if isapi_frame is not None:
                        source_frame = isapi_frame
                        # Re-detect face in ISAPI frame for accurate crop
                        isapi_dets = self.detector.detect(source_frame)
                        if isapi_dets:
                            isapi_face = self._pick_center_largest(
                                isapi_dets, source_frame.shape)
                            crop = self._crop_expand(source_frame, isapi_face.bbox)
                        else:
                            # Fallback: crop same region from ISAPI frame
                            crop = self._crop_expand(source_frame, face.bbox)
                    else:
                        crop = self._crop_expand(frame, face.bbox)
                else:
                    crop = self._crop_expand(frame, face.bbox)
                if crop is not None and quality_ok(crop):
                    return CaptureAction("collect", face_crop=crop)
                return CaptureAction("none")
            else:
                # Danger zone: trigger correction
                return self._trigger_correction(face.bbox)

        else:
            # Face lost
            lost_ms = (now - self.last_face_ts) * 1000

            if lost_ms < self.cfg.face_lost_kalman_ms:
                pred_cx, pred_cy = self._kalman_predict()
                pred_bbox = self._make_bbox_from_center(
                    pred_cx, pred_cy, self.last_face_bbox, frame.shape)
                if not self._in_safe_zone(pred_cx, pred_cy, frame.shape):
                    return self._trigger_correction(pred_bbox)
                return CaptureAction("none")

            elif lost_ms < self.cfg.face_lost_giveup_ms:
                if self.correction_count < self.cfg.max_corrections:
                    pred_cx, pred_cy = self._kalman_predict()
                    pred_bbox = self._make_bbox_from_center(
                        pred_cx, pred_cy, self.last_face_bbox, frame.shape)
                    return self._trigger_correction(pred_bbox)
                return CaptureAction("none")
            else:
                return CaptureAction("giveup")

    def enter_correction_settle(self):
        """Called by state_machine after 3D re-positioning."""
        self.in_correction_settle = True
        self.settle_until = time.time() + self.cfg.correction_settle

    def _trigger_correction(self, bbox) -> CaptureAction:
        if self.correction_count >= self.cfg.max_corrections:
            return CaptureAction("giveup")
        self.correction_count += 1
        return CaptureAction("correct", corrected_bbox=bbox)

    def _in_safe_zone(self, cx, cy, frame_shape) -> bool:
        h, w = frame_shape[:2]
        margin_x = w * (1 - self.cfg.safe_zone_ratio) / 2
        margin_y = h * (1 - self.cfg.safe_zone_ratio) / 2
        return (margin_x <= cx <= w - margin_x and
                margin_y <= cy <= h - margin_y)

    def _pick_center_largest(self, dets, frame_shape):
        """Pick face closest to center, breaking ties by area."""
        h, w = frame_shape[:2]
        return min(dets, key=lambda d: (
            ((d.bbox[0] + d.bbox[2]) / 2 - w / 2) ** 2 +
            ((d.bbox[1] + d.bbox[3]) / 2 - h / 2) ** 2
        ))

    def _crop_expand(self, frame, bbox, expand=1.3):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * expand, (y2 - y1) * expand
        h, w = frame.shape[:2]
        x1e = max(int(cx - bw / 2), 0)
        y1e = max(int(cy - bh / 2), 0)
        x2e = min(int(cx + bw / 2), w)
        y2e = min(int(cy + bh / 2), h)
        crop = frame[y1e:y2e, x1e:x2e]
        return crop if crop.size > 0 else None

    def _init_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1],
             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        return kf

    def _kalman_update(self, cx, cy):
        m = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.kf_initialized:
            self.kf.statePre = np.array(
                [[cx], [cy], [0], [0]], np.float32)
            self.kf_initialized = True
        self.kf.correct(m)
        self.kf.predict()

    def _kalman_predict(self) -> Tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def _make_bbox_from_center(self, cx, cy, ref_bbox, frame_shape):
        if ref_bbox is None:
            return (0, 0, frame_shape[1], frame_shape[0])
        rw = (ref_bbox[2] - ref_bbox[0]) / 2
        rh = (ref_bbox[3] - ref_bbox[1]) / 2
        h, w = frame_shape[:2]
        return (max(int(cx - rw), 0), max(int(cy - rh), 0),
                min(int(cx + rw), w), min(int(cy + rh), h))
