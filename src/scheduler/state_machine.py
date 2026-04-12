"""Scan scheduler state machine: PATROL + SCAN modes."""

import logging
import time
from collections import deque
from enum import IntEnum
from typing import List, Optional, Deque

import cv2
import numpy as np

from src.config import AppConfig
from src.sdk.hik_ptz import HikPTZ
from src.video.rtsp_source import RtspSource
from src.detect.yolo_face import YoloFace, Detection
from src.detect.yolo_person import YoloPerson

logger = logging.getLogger("app")


class State(IntEnum):
    INIT = 0
    PATROL_GOTO = 1
    PATROL_DWELL = 2
    SCAN_GOTO_PRESET = 3
    SCAN_DETECT = 4
    SCAN_PICK = 5
    SCAN_ZOOM_IN = 6
    SCAN_SETTLE = 7
    SCAN_CAPTURE = 8
    SCAN_RECOGNIZE = 9
    SCAN_ZOOM_OUT = 10
    SCAN_NEXT_PRESET = 11


class ScanScheduler:
    """Main state machine driving patrol and scan modes."""

    def __init__(self, cfg: AppConfig, ptz: HikPTZ, video: RtspSource,
                 face_wide: YoloFace, face_close: YoloFace,
                 person_det: Optional[YoloPerson] = None):
        self.cfg = cfg
        self.ptz = ptz
        self.video = video
        self.face_wide = face_wide
        self.face_close = face_close
        self.person_det = person_det

        self.state = State.INIT
        self.stop_flag = False
        self.is_recording = False
        self.is_paused = False

        # Patrol state
        self.patrol_idx = 0
        self.settle_deadline = 0.0
        self.dwell_deadline: Optional[float] = None
        self._prev_gray: Optional[np.ndarray] = None

        # Scan state
        self.scan_preset_queue: Deque[int] = deque()
        self.current_scan_preset: Optional[int] = None
        self.target = None  # Current target being scanned
        self.capture_buf: List[np.ndarray] = []
        self.capture_deadline = 0.0

        # Face confirmation (for patrol)
        self._face_confirm_count = 0

        # FPS tracking
        self._fps_ts = time.time()
        self._fps_count = 0
        self.fps = 0.0

    def run(self):
        """Main state machine loop."""
        logger.info("ScanScheduler starting")

        while not self.stop_flag:
            frame = self.video.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Update FPS
            self._fps_count += 1
            now = time.time()
            if now - self._fps_ts >= 1.0:
                self.fps = self._fps_count / (now - self._fps_ts)
                self._fps_count = 0
                self._fps_ts = now

            if self.is_paused:
                time.sleep(0.05)
                continue

            # State dispatch
            if self.state == State.INIT:
                self._handle_init()
            elif self.state == State.PATROL_GOTO:
                self._handle_patrol_goto()
            elif self.state == State.PATROL_DWELL:
                self._handle_patrol_dwell(frame)
            elif self.state == State.SCAN_GOTO_PRESET:
                self._handle_scan_goto_preset()
            elif self.state == State.SCAN_DETECT:
                self._handle_scan_detect(frame)
            elif self.state == State.SCAN_PICK:
                self._handle_scan_pick(frame)
            elif self.state == State.SCAN_ZOOM_IN:
                self._handle_scan_zoom_in(frame)
            elif self.state == State.SCAN_SETTLE:
                self._handle_scan_settle(frame)
            elif self.state == State.SCAN_CAPTURE:
                self._handle_scan_capture(frame)
            elif self.state == State.SCAN_RECOGNIZE:
                self._handle_scan_recognize()
            elif self.state == State.SCAN_ZOOM_OUT:
                self._handle_scan_zoom_out(frame)
            elif self.state == State.SCAN_NEXT_PRESET:
                self._handle_scan_next_preset()

        logger.info("ScanScheduler stopped")

    # ── State handlers ──

    def _handle_init(self):
        """Initialization complete, enter patrol."""
        logger.info("INIT -> PATROL_GOTO")
        self.state = State.PATROL_GOTO

    def _handle_patrol_goto(self):
        """Move to next patrol preset."""
        preset = self.cfg.patrol.presets[self.patrol_idx]
        logger.info(f"PATROL_GOTO: moving to preset {preset}")
        self.ptz.goto_preset(preset)
        self.settle_deadline = time.time() + self.cfg.ptz.settle_timeout
        self.dwell_deadline = None
        self._prev_gray = None
        self.state = State.PATROL_DWELL

    def _handle_patrol_dwell(self, frame):
        """Stay at preset, detect faces."""
        # Wait for settle
        if not self._frame_settled(frame) and time.time() < self.settle_deadline:
            return

        if self.dwell_deadline is None:
            self.dwell_deadline = time.time() + self.cfg.patrol.dwell
            self._face_confirm_count = 0
            logger.info(
                f"PATROL_DWELL: settled at preset "
                f"{self.cfg.patrol.presets[self.patrol_idx]}"
            )

        # Detect faces
        dets = self.face_wide.detect(frame)

        if len(dets) > 0:
            self._face_confirm_count += 1
            logger.debug(
                f"  Detected {len(dets)} faces "
                f"(confirm: {self._face_confirm_count}/{self.cfg.patrol.min_confirm_frames})"
            )
            if self._face_confirm_count >= self.cfg.patrol.min_confirm_frames:
                # Found people -> transition to SCAN
                trigger_preset = self.cfg.patrol.presets[self.patrol_idx]
                self._build_scan_order(trigger_preset)
                logger.info(
                    f"Face confirmed at preset {trigger_preset}, "
                    f"entering SCAN mode"
                )
                self.state = State.SCAN_GOTO_PRESET
                return
        else:
            self._face_confirm_count = 0

        # Dwell timeout -> next preset
        if time.time() > self.dwell_deadline:
            self.patrol_idx = (
                (self.patrol_idx + 1) % len(self.cfg.patrol.presets)
            )
            self.state = State.PATROL_GOTO

    def _handle_scan_goto_preset(self):
        """Move to scan preset."""
        if self.current_scan_preset is None:
            if self.scan_preset_queue:
                self.current_scan_preset = self.scan_preset_queue.popleft()
            else:
                self.state = State.PATROL_GOTO
                return

        logger.info(f"SCAN_GOTO_PRESET: moving to preset {self.current_scan_preset}")
        self.ptz.goto_preset(self.current_scan_preset)
        self.settle_deadline = time.time() + self.cfg.ptz.settle_timeout
        self._prev_gray = None
        self.state = State.SCAN_DETECT

    def _handle_scan_detect(self, frame):
        """Detect faces and build scan queue."""
        if not self._frame_settled(frame) and time.time() < self.settle_deadline:
            return

        logger.info("SCAN_DETECT: running detection")
        dets = self.face_wide.detect(frame)

        # Sort by area descending, then left-to-right
        dets.sort(key=lambda d: (
            -(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            d.bbox[0],
        ))

        self._scan_queue = dets
        logger.info(f"  Found {len(dets)} faces in scan queue")
        self.state = State.SCAN_PICK

    def _handle_scan_pick(self, frame):
        """Pick next target from scan queue."""
        if not hasattr(self, '_scan_queue') or not self._scan_queue:
            self.state = State.SCAN_NEXT_PRESET
            return

        self.target = self._scan_queue.pop(0)
        logger.info(f"SCAN_PICK: targeting face at {self.target.bbox}")

        # Zoom to target
        ok = self.ptz.zoom_to_bbox(
            self.target.bbox, frame.shape[1], frame.shape[0],
            expand=self.cfg.ptz.expand_ratio,
        )
        if ok:
            self.settle_deadline = time.time() + self.cfg.ptz.settle_timeout
            self._prev_gray = None
            self.state = State.SCAN_SETTLE
        else:
            logger.warning("zoom_to_bbox failed, skipping target")
            self.target = None
            self.state = State.SCAN_PICK

    def _handle_scan_zoom_in(self, frame):
        """Already handled in SCAN_PICK."""
        self.state = State.SCAN_SETTLE

    def _handle_scan_settle(self, frame):
        """Wait for frame to settle after zoom."""
        if not self._frame_settled(frame) and time.time() < self.settle_deadline:
            return

        logger.info("SCAN_SETTLE: frame settled, entering CAPTURE")
        self.capture_buf = []
        self.capture_deadline = time.time() + self.cfg.capture.timeout
        self.state = State.SCAN_CAPTURE

    def _handle_scan_capture(self, frame):
        """Capture face images. (Basic version without CaptureTracker)"""
        if time.time() > self.capture_deadline:
            logger.info("SCAN_CAPTURE: timeout")
            self.state = State.SCAN_RECOGNIZE
            return

        dets = self.face_close.detect(frame)
        if not dets:
            return

        # Pick center-most and largest
        fh, fw = frame.shape[:2]
        best = min(dets, key=lambda d: (
            ((d.bbox[0] + d.bbox[2]) / 2 - fw / 2) ** 2 +
            ((d.bbox[1] + d.bbox[3]) / 2 - fh / 2) ** 2
        ))

        x1, y1, x2, y2 = best.bbox
        # Expand slightly
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * 1.3, (y2 - y1) * 1.3
        x1e = max(int(cx - bw / 2), 0)
        y1e = max(int(cy - bh / 2), 0)
        x2e = min(int(cx + bw / 2), fw)
        y2e = min(int(cy + bh / 2), fh)

        crop = frame[y1e:y2e, x1e:x2e]
        if crop.size > 0:
            from src.utils.quality import quality_ok
            if quality_ok(crop):
                self.capture_buf.append(crop)
                logger.debug(f"  Captured face #{len(self.capture_buf)}")

        if len(self.capture_buf) >= self.cfg.capture.max_samples:
            self.state = State.SCAN_RECOGNIZE

    def _handle_scan_recognize(self):
        """Recognize captured faces. Placeholder - will be enhanced in M5."""
        logger.info(
            f"SCAN_RECOGNIZE: {len(self.capture_buf)} captures"
        )
        self.target = None
        self.state = State.SCAN_ZOOM_OUT

    def _handle_scan_zoom_out(self, frame):
        """Return to current scan preset."""
        if self.current_scan_preset is not None:
            self.ptz.goto_preset(self.current_scan_preset)
        self.settle_deadline = time.time() + self.cfg.ptz.settle_timeout
        self._prev_gray = None
        self.state = State.SCAN_PICK

    def _handle_scan_next_preset(self):
        """Move to next preset in scan queue."""
        if self.scan_preset_queue:
            self.current_scan_preset = self.scan_preset_queue.popleft()
            logger.info(f"SCAN_NEXT_PRESET: moving to {self.current_scan_preset}")
            self.state = State.SCAN_GOTO_PRESET
        else:
            logger.info("All presets scanned, returning to PATROL")
            self.current_scan_preset = None
            self.state = State.PATROL_GOTO

    # ── Helpers ──

    def _frame_settled(self, frame) -> bool:
        """Check if frame motion is below threshold (frame diff)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False
        diff = cv2.absdiff(gray, self._prev_gray).mean()
        self._prev_gray = gray
        return diff < self.cfg.ptz.settle_diff_th

    def _build_scan_order(self, trigger_preset: int):
        """Build scan preset order starting from trigger preset."""
        presets = self.cfg.patrol.presets
        try:
            trigger_idx = presets.index(trigger_preset)
        except ValueError:
            trigger_idx = 0

        self.scan_preset_queue.clear()
        for i in range(len(presets)):
            self.scan_preset_queue.append(
                presets[(trigger_idx + i) % len(presets)]
            )
        self.current_scan_preset = self.scan_preset_queue.popleft()

    def handle_command(self, cmd: str):
        """Handle user commands."""
        if cmd in ("q", "quit"):
            logger.info("Quit command received")
            self.stop_flag = True
        elif cmd in ("p", "pause"):
            self.is_paused = not self.is_paused
            logger.info(f"{'Paused' if self.is_paused else 'Resumed'}")
        elif cmd in ("h", "home"):
            self.ptz.goto_preset(1)
            self.state = State.PATROL_GOTO
            self.patrol_idx = 0
            logger.info("Reset to PATROL_GOTO preset 1")
        elif cmd in ("r", "reset"):
            self.state = State.PATROL_GOTO
            self.patrol_idx = 0
            logger.info("Reset to PATROL")
