"""SORT + ReID tracker for cross-frame and cross-preset identity maintenance."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import TrackConfig
from src.detect.yolo_face import Detection
from src.reid.osnet import OSNetReID
from src.utils.geometry import iou, cosine_similarity, bbox_center

logger = logging.getLogger("app")


@dataclass
class Track:
    """Tracked person/face identity."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    face_feat: Optional[np.ndarray] = None     # ArcFace 512-d
    reid_feat: Optional[np.ndarray] = None      # OSNet 512-d
    last_seen_ts: float = 0.0
    age: int = 0
    hits: int = 0

    @property
    def area(self):
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


class Tracker:
    """SORT-style tracker with ReID feature fusion.

    Uses Hungarian matching with combined IoU + ReID distance.
    """

    def __init__(self, cfg: TrackConfig, reid: Optional[OSNetReID] = None):
        self.cfg = cfg
        self.reid = reid
        self.tracks: List[Track] = []
        self._next_id = 1
        self._snapshot: List[Tuple[int, np.ndarray]] = []  # (track_id, reid_feat)

    def update(self, frame, face_dets: List[Detection],
               person_dets: Optional[List[Detection]] = None,
               timestamp: float = 0.0):
        """Update tracker with new detections.

        Associates face detections with existing tracks using IoU + ReID cost.
        """
        if not face_dets:
            # Age all tracks
            for t in self.tracks:
                t.age += 1
            self.tracks = [t for t in self.tracks
                           if t.age <= self.cfg.max_age]
            return

        # If no existing tracks, create new ones
        if not self.tracks:
            for det in face_dets:
                self._create_track(det, frame, timestamp)
            return

        # Build cost matrix
        n_tracks = len(self.tracks)
        n_dets = len(face_dets)
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(face_dets):
                iou_cost = 1.0 - iou(track.bbox, det.bbox)

                reid_cost = 1.0  # Default if no ReID
                if (self.reid is not None and
                        track.reid_feat is not None):
                    # Extract ReID feat for detection
                    det_reid = self._extract_reid(frame, det)
                    if det_reid is not None:
                        reid_cost = 1.0 - cosine_similarity(
                            track.reid_feat, det_reid)

                cost[i, j] = (self.cfg.iou_weight * iou_cost +
                              self.cfg.reid_weight * reid_cost)

        # Hungarian matching
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < 1.0:  # Threshold
                self.tracks[r].bbox = face_dets[c].bbox
                self.tracks[r].last_seen_ts = timestamp
                self.tracks[r].hits += 1
                self.tracks[r].age = 0
                # Update ReID feature (EMA)
                if self.reid is not None:
                    det_reid = self._extract_reid(frame, face_dets[c])
                    if det_reid is not None:
                        if self.tracks[r].reid_feat is None:
                            self.tracks[r].reid_feat = det_reid
                        else:
                            alpha = self.cfg.reid_ema
                            self.tracks[r].reid_feat = (
                                (1 - alpha) * self.tracks[r].reid_feat +
                                alpha * det_reid
                            )
                            norm = np.linalg.norm(self.tracks[r].reid_feat)
                            if norm > 1e-9:
                                self.tracks[r].reid_feat /= norm

                matched_tracks.add(r)
                matched_dets.add(c)

        # Create new tracks for unmatched detections
        for j in range(n_dets):
            if j not in matched_dets:
                self._create_track(face_dets[j], frame, timestamp)

        # Age unmatched tracks
        for i in range(n_tracks):
            if i not in matched_tracks:
                self.tracks[i].age += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks
                       if t.age <= self.cfg.max_age]

    def snapshot_before_move(self):
        """Save current track ReID features for cross-zoom recovery."""
        self._snapshot = [
            (t.track_id, t.reid_feat.copy()) for t in self.tracks
            if t.reid_feat is not None
        ]

    def restore_after_move(self, new_dets: List[Detection],
                           frame: np.ndarray, timestamp: float = 0.0):
        """Restore track IDs after PTZ move using ReID matching."""
        if not self._snapshot or not new_dets or self.reid is None:
            self.tracks.clear()
            for det in new_dets:
                self._create_track(det, frame, timestamp)
            return

        for det in new_dets:
            det_reid = self._extract_reid(frame, det)
            best_id = None
            best_sim = 0.0

            if det_reid is not None:
                for snap_id, snap_feat in self._snapshot:
                    sim = cosine_similarity(det_reid, snap_feat)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = snap_id

            track = Track(
                track_id=best_id if best_sim > 0.5 else self._next_id,
                bbox=det.bbox,
                reid_feat=det_reid,
                last_seen_ts=timestamp,
                hits=1,
            )
            if track.track_id == self._next_id:
                self._next_id += 1
            self.tracks.append(track)

    def reset(self):
        """Reset all tracks."""
        self.tracks.clear()
        self._snapshot.clear()

    def _create_track(self, det: Detection, frame, timestamp: float):
        """Create a new track from detection."""
        track = Track(
            track_id=self._next_id,
            bbox=det.bbox,
            last_seen_ts=timestamp,
            hits=1,
        )
        if self.reid is not None:
            track.reid_feat = self._extract_reid(frame, det)
        self._next_id += 1
        self.tracks.append(track)

    def _extract_reid(self, frame, det: Detection) -> Optional[np.ndarray]:
        """Extract ReID feature from detection region."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox
        # Expand to get more person context
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        expand = 2.0
        nx1 = max(int(cx - bw * expand / 2), 0)
        ny1 = max(int(cy - bh * expand / 2), 0)
        nx2 = min(int(cx + bw * expand / 2), w)
        ny2 = min(int(cy + bh * expand / 2), h)

        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
            return None
        return self.reid.embed(crop)
