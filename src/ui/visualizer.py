"""Frame annotation renderer - pure CPU OpenCV drawing, no display dependency."""

import time
from typing import List, Optional

import cv2
import numpy as np

# Color constants (BGR)
COLOR_HIT = (0, 255, 0)        # Green
COLOR_STRANGER = (0, 0, 255)   # Red
COLOR_PENDING = (0, 255, 255)  # Yellow
COLOR_SCANNING = (255, 128, 0) # Orange
COLOR_UNKNOWN = (180, 180, 180) # Gray


class Visualizer:
    """Renders annotations on frames without displaying."""

    def __init__(self):
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, tracks=None,
               state=None, preset_id: Optional[int] = None,
               queue_size: int = 0, fps: float = 0.0,
               is_recording: bool = False,
               identified_names: Optional[List[str]] = None) -> np.ndarray:
        """Draw all annotations on frame copy and return it."""
        annotated = frame.copy()

        # Draw tracks
        if tracks:
            for track in tracks:
                self._draw_track(annotated, track)

        # HUD - top left
        self._draw_hud(annotated, state, preset_id, queue_size,
                        fps, is_recording)

        # Identified names panel - right side
        if identified_names:
            self._draw_identified_panel(annotated, identified_names)

        return annotated

    def _draw_track(self, frame, track):
        """Draw a single track bbox with label."""
        if track.bbox is None:
            return

        x1, y1, x2, y2 = track.bbox
        color = COLOR_PENDING
        label = f"#{track.track_id}"

        # Use track attributes to determine color/label
        if hasattr(track, 'scan_result') and track.scan_result:
            if track.scan_result == "STRANGER":
                color = COLOR_STRANGER
                label = f"#{track.track_id} STRANGER"
            elif track.scan_result not in ("UNKNOWN", ""):
                color = COLOR_HIT
                label = f"#{track.track_id} {track.scan_result}"
        if hasattr(track, 'hits') and track.hits < 3:
            color = COLOR_PENDING

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    self._font, 0.6, color, 2)

    def _draw_hud(self, frame, state, preset_id, queue_size, fps,
                  is_recording):
        """Draw HUD overlay on top-left."""
        lines = []
        if state is not None:
            state_name = state.name if hasattr(state, 'name') else str(state)
            lines.append(f"State: {state_name}")
        if preset_id is not None:
            lines.append(f"Preset: {preset_id}")
        lines.append(f"Queue: {queue_size}")
        lines.append(f"FPS: {fps:.1f}")
        if is_recording:
            lines.append("REC")

        y = 30
        for line in lines:
            color = (0, 0, 255) if line == "REC" else (0, 255, 255)
            cv2.putText(frame, line, (10, y), self._font, 0.7, color, 2)
            y += 28

        # REC indicator - blinking effect
        if is_recording:
            t = time.time()
            if int(t * 2) % 2 == 0:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12,
                           (0, 0, 255), -1)
                cv2.putText(frame, "REC", (frame.shape[1] - 80, 35),
                            self._font, 0.5, (0, 0, 255), 2)

    def _draw_identified_panel(self, frame, names):
        """Draw identified persons panel on right side."""
        if not names:
            return

        x = frame.shape[1] - 200
        y = 30
        cv2.putText(frame, "Identified:", (x, y), self._font, 0.5,
                    (255, 255, 255), 1)
        y += 25
        for name in names[-10:]:  # Show last 10
            cv2.putText(frame, name, (x, y), self._font, 0.4,
                        COLOR_HIT, 1)
            y += 20
