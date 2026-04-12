"""Scan scheduler state machine: PATROL + SCAN modes."""

import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from enum import IntEnum
from typing import List, Optional, Deque

import cv2
import numpy as np

from src.config import AppConfig
from src.sdk.hik_ptz import HikPTZ
from src.video.rtsp_source import RtspSource
from src.detect.yolo_face import YoloFace, Detection
from src.detect.yolo_person import YoloPerson
from src.recognize.arcface import ArcFace
from src.recognize.gallery import FaceGallery, MatchResult
from src.scheduler.capture_tracker import CaptureTracker
from src.reid.osnet import OSNetReID
from src.track.sort_reid import Tracker, Track
from src.ui.visualizer import Visualizer
from src.ui.display import DisplayBackend
from src.utils.geometry import iou

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
                 person_det: Optional[YoloPerson] = None,
                 arcface: Optional[ArcFace] = None,
                 gallery: Optional[FaceGallery] = None,
                 reid: Optional[OSNetReID] = None):
        self.cfg = cfg
        self.ptz = ptz
        self.video = video
        self.face_wide = face_wide
        self.face_close = face_close
        self.person_det = person_det
        self.arcface = arcface
        self.gallery = gallery

        # Tracker + ReID
        self.reid = reid
        self.tracker: Optional[Tracker] = None
        if reid is not None:
            self.tracker = Tracker(cfg.track, reid)

        # Global identified persons table (cross-preset dedup)
        self.identified: List[dict] = []  # {track_id, reid_feat, result, name, sim}

        # Visualization
        self.visualizer = Visualizer()
        self.display: Optional[DisplayBackend] = None

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
        self.target = None  # Current target Detection being scanned
        self.target_bbox = None  # bbox of target
        self.capture_buf: List[np.ndarray] = []
        self.capture_deadline = 0.0
        self.capture_tracker: Optional[CaptureTracker] = None

        # Event log
        self._event_count = 0

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

            # Render and display
            self._render_frame(frame)

            # Poll commands
            cmd = self.display.poll_command() if self.display else None
            if cmd:
                self.handle_command(cmd)

        logger.info("ScanScheduler stopped")

    # ── State handlers ──

    def _handle_init(self):
        """Initialization complete, enter patrol."""
        if self.display is None:
            self.display = DisplayBackend(
                mode=self.cfg.display.mode,
                web_host=self.cfg.display.web_host,
                web_port=self.cfg.display.web_port,
                jpeg_quality=self.cfg.display.jpeg_quality,
            )
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
        """Detect faces, run tracker, do cross-preset dedup."""
        if not self._frame_settled(frame) and time.time() < self.settle_deadline:
            return

        logger.info("SCAN_DETECT: running detection")
        dets = self.face_wide.detect(frame)

        # Run tracker
        now = time.time()
        if self.tracker is not None:
            self.tracker.update(frame, dets, timestamp=now)

            # Cross-preset dedup: check each track against identified table
            scan_dets = []
            for track in self.tracker.tracks:
                if track.reid_feat is not None:
                    matched = self._is_already_identified(track.reid_feat)
                    if matched:
                        logger.info(
                            f"  Track {track.track_id} already identified "
                            f"as '{matched['name']}', skipping"
                        )
                        continue
                # Find corresponding detection
                for det in dets:
                    if self._bbox_overlap(det.bbox, track.bbox) > 0.3:
                        scan_dets.append(det)
                        break
        else:
            scan_dets = dets

        # Sort by area descending, then left-to-right
        scan_dets.sort(key=lambda d: (
            -(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            d.bbox[0],
        ))

        self._scan_queue = scan_dets
        logger.info(f"  {len(dets)} detected, {len(scan_dets)} to scan after dedup")
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
        self.capture_tracker = None
        if self.target is not None:
            self.target_bbox = self.target.bbox
        self.state = State.SCAN_CAPTURE

    def _handle_scan_capture(self, frame):
        """Capture face images with CaptureTracker dynamic correction."""
        if time.time() > self.capture_deadline:
            logger.info("SCAN_CAPTURE: timeout")
            self.state = State.SCAN_RECOGNIZE
            return

        # Initialize capture tracker on first frame
        if self.capture_tracker is None:
            self.capture_tracker = CaptureTracker(
                self.face_close, self.cfg.capture.tracking,
            )
            # Prime it with the target bbox if available
            if self.target_bbox is not None:
                self.capture_tracker.last_face_bbox = self.target_bbox

        action = self.capture_tracker.step(frame)

        if action.type == "collect":
            self.capture_buf.append(action.face_crop)
            logger.debug(f"  Captured face #{len(self.capture_buf)}")
        elif action.type == "correct":
            # Layer B: 3D re-positioning
            logger.info(f"  Correction #{self.capture_tracker.correction_count}")
            ok = self.ptz.zoom_to_bbox(
                action.corrected_bbox, frame.shape[1], frame.shape[0],
                expand=self.cfg.ptz.expand_ratio,
            )
            if ok:
                self.capture_tracker.enter_correction_settle()
            else:
                logger.warning("  Correction zoom failed")
        elif action.type == "giveup":
            logger.info("SCAN_CAPTURE: target lost, giving up")
            self.state = State.SCAN_ZOOM_OUT
            return

        if len(self.capture_buf) >= self.cfg.capture.min_samples:
            self.state = State.SCAN_RECOGNIZE

    def _handle_scan_recognize(self):
        """Recognize captured faces using ArcFace + Gallery."""
        logger.info(f"SCAN_RECOGNIZE: {len(self.capture_buf)} captures")

        if self.capture_buf and self.arcface and self.gallery:
            # Embed all captures
            feats = [self.arcface.embed(f) for f in self.capture_buf]
            mean_feat = np.mean(feats, axis=0)
            mean_feat /= (np.linalg.norm(mean_feat) + 1e-9)

            # Match against gallery
            result = self.gallery.match(
                mean_feat,
                match_th=self.cfg.recognize.match_th,
                reject_th=self.cfg.recognize.reject_th,
            )

            # Determine final result
            if result.kind == "hit":
                name = result.name
                logger.info(f"  HIT: {name} (sim={result.sim:.4f})")
            elif result.kind == "stranger":
                name = "STRANGER"
                logger.info(f"  STRANGER (best_sim={result.sim:.4f})")
                # Save stranger snapshot
                self._save_stranger_snapshot()
            else:
                # Ambiguous - use top1 as fallback
                name = result.name or "UNKNOWN"
                logger.info(f"  AMBIGUOUS: {name} (sim={result.sim:.4f})")

            # Log event
            self._log_event(result, name)

            # Add to identified table
            self._add_identified(result, name)
        else:
            logger.info("  No captures or recognizer not available")

        # Reset capture state
        self.capture_tracker = None
        self.target = None
        self.target_bbox = None
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
            if self.tracker is not None:
                self.tracker.snapshot_before_move()
            logger.info(f"SCAN_NEXT_PRESET: moving to {self.current_scan_preset}")
            self.state = State.SCAN_GOTO_PRESET
        else:
            logger.info("All presets scanned, returning to PATROL")
            self.current_scan_preset = None
            self.identified.clear()
            if self.tracker is not None:
                self.tracker.reset()
            self.state = State.PATROL_GOTO

    def _log_event(self, result: MatchResult, name: str):
        """Append recognition event to events.jsonl."""
        os.makedirs(os.path.dirname(self.cfg.output.events_jsonl) or ".", exist_ok=True)
        self._event_count += 1
        event = {
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "preset": self.current_scan_preset,
            "event_id": self._event_count,
            "result": result.kind,
            "name": name,
            "sim": round(result.sim, 4),
        }
        if self.target_bbox:
            event["bbox"] = list(self.target_bbox)

        with open(self.cfg.output.events_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_stranger_snapshot(self):
        """Save best capture frame for stranger."""
        if not self.capture_buf:
            return
        os.makedirs(self.cfg.output.strangers_dir, exist_ok=True)
        # Save the last (hopefully best quality) capture
        best = max(self.capture_buf, key=lambda c: c.shape[0])
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(
            self.cfg.output.strangers_dir,
            f"{ts}-{self._event_count}.jpg",
        )
        cv2.imwrite(path, best)
        logger.info(f"  Stranger snapshot saved: {path}")

    def _render_frame(self, frame):
        """Render annotations and push to display."""
        tracks = self.tracker.tracks if self.tracker else []
        identified_names = [p["name"] for p in self.identified]

        annotated = self.visualizer.render(
            frame, tracks=tracks, state=self.state,
            preset_id=self.current_scan_preset,
            queue_size=len(getattr(self, '_scan_queue', [])),
            fps=self.fps,
            is_recording=self.is_recording,
            identified_names=identified_names,
        )
        if self.display:
            self.display.show(annotated)

    def _is_already_identified(self, reid_feat: np.ndarray) -> Optional[dict]:
        """Check if a ReID feature matches anyone in the identified table."""
        threshold = self.cfg.reid.cross_preset_th
        best = None
        best_sim = 0.0
        for person in self.identified:
            sim = float(np.dot(reid_feat, person["reid_feat"]))
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best = person
        return best

    def _add_identified(self, result: MatchResult, name: str):
        """Add recognized person to global identified table."""
        # Get reid_feat from tracker if available
        reid_feat = None
        if self.tracker is not None:
            for track in self.tracker.tracks:
                if track.reid_feat is not None:
                    reid_feat = track.reid_feat
                    break

        if reid_feat is not None:
            self.identified.append({
                "name": name,
                "reid_feat": reid_feat,
                "result": result.kind,
                "sim": result.sim,
            })

    def _bbox_overlap(self, a, b) -> float:
        """Quick IoU check between two bboxes."""
        return iou(a, b)

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
        elif cmd in ("v", "record"):
            if self.is_recording:
                self.ptz.stop_record()
                self.is_recording = False
                logger.info("Recording stopped")
            else:
                ok = self.ptz.start_record()
                if ok:
                    self.is_recording = True
                    logger.info("Recording started")
