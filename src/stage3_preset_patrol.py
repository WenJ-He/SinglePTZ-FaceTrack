"""Stage 3: preset patrol with observing, simple tracking, and reset-to-nearest-preset."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import argparse
import logging
import signal
import time

from src.app_config import auto_providers, load_app_config
from src.detector import FaceDetector, PersonDetector
from src.geometry import bbox_center
from src.hik_sdk import HikClient
from src.logger import setup_logger
from src.ptz_controller import HikManualPtzController
from src.ptz_position import PtzCoord, PtzScope, capture_preset_coords, get_ptz_coord, get_ptz_scope, nearest_preset
from src.stage1_single_static import Stage1SingleStaticApp
from src.tracker import SingleTargetTracker
from src.video_source import RtspVideoSource
from src.web import Stage1Overlay, WebDebugServer

logger = logging.getLogger("app")


class Stage3State(str, Enum):
    PATROLLING = "PATROLLING"
    OBSERVING = "OBSERVING"
    TRACKING = "TRACKING"
    RESETTING = "RESETTING"


@dataclass
class TargetObservation:
    ts: float
    x: float
    y: float
    bbox: tuple[int, int, int, int]


class Stage3PresetPatrolApp(Stage1SingleStaticApp):
    """Outer preset patrol state machine with a simple Stage2-like tracking loop."""

    def __init__(self, common_config: str, stage_config: str, enable_web: bool = True, timing_log_interval_s: float = 1.0):
        super().__init__(common_config, stage_config, enable_web=enable_web, timing_log_interval_s=timing_log_interval_s)
        self.state = Stage3State.PATROLLING
        self.person_tracker = SingleTargetTracker(lost_timeout_s=self.cfg.stage3.lost_all_reset_s)
        self.face_tracker = SingleTargetTracker(lost_timeout_s=self.cfg.stage3.lost_face_to_body_s)
        self.overlay = Stage1Overlay()

        self.face_history = deque(maxlen=6)
        self.body_history = deque(maxlen=6)
        self.last_face_seen_ts = 0.0
        self.last_body_seen_ts = 0.0

        self.track_until = 0.0
        self.track_started_wall_ts = 0.0
        self.track_lost_since = 0.0
        self.observe_until = 0.0
        self.current_preset_idx = -1
        self.next_preset_idx = 0
        self.preset_coords: dict[int, PtzCoord] = {}
        self.ptz_scope: PtzScope | None = None
        self.last_known_ptz: PtzCoord | None = None
        self.face_zoom_ready_count = 0
        self.tilt_up_blocked_until = 0.0

        self.client = None

    def _set_state(self, new_state: Stage3State, reason: str = ""):
        if self.state != new_state:
            suffix = f" ({reason})" if reason else ""
            logger.info("STATE %s -> %s%s", self.state.value, new_state.value, suffix)
        self.state = new_state

    def _current_preset_id(self) -> int | None:
        if 0 <= self.current_preset_idx < len(self.cfg.stage3.preset_ids):
            return self.cfg.stage3.preset_ids[self.current_preset_idx]
        return None

    def _next_index(self) -> int:
        if not self.cfg.stage3.preset_ids:
            return 0
        if self.current_preset_idx < 0:
            return 0
        return (self.current_preset_idx + 1) % len(self.cfg.stage3.preset_ids)

    def _clear_tracking_context(self):
        self.person_tracker.reset()
        self.face_tracker.reset()
        self.face_history.clear()
        self.body_history.clear()
        self.last_face_seen_ts = 0.0
        self.last_body_seen_ts = 0.0
        self.track_until = 0.0
        self.track_started_wall_ts = 0.0
        self.track_lost_since = 0.0
        self.face_zoom_ready_count = 0
        self.tilt_up_blocked_until = 0.0
        self.zoom_steps = 0
        self.last_zoom_ts = 0.0
        self.settling_until = 0.0

    def _initialize_preset_map(self):
        self.ptz_scope, scope_channel = get_ptz_scope(self.client, self.cfg.hik.channel)
        if self.ptz_scope is None:
            logger.warning("PTZ scope unavailable, nearest-preset distance will use fallback ranges")
        else:
            logger.info(
                "[PTZ] scope channel=%d pan=[%.1f, %.1f] tilt=[%.1f, %.1f] zoom=[%.1f, %.1f]",
                scope_channel,
                self.ptz_scope.pan_min,
                self.ptz_scope.pan_max,
                self.ptz_scope.tilt_min,
                self.ptz_scope.tilt_max,
                self.ptz_scope.zoom_min,
                self.ptz_scope.zoom_max,
            )

        logger.info("[PTZ] capturing preset coordinates for ids=%s", self.cfg.stage3.preset_ids)
        self.preset_coords = capture_preset_coords(
            self.client,
            self.cfg.stage3.preset_ids,
            self.cfg.stage3.preset_capture_settle_s,
            self.cfg.hik.channel,
        )
        for preset_id in self.cfg.stage3.preset_ids:
            coord = self.preset_coords[preset_id]
            logger.info(
                "[PTZ] preset=%d pan=%.1f tilt=%.1f zoom=%.1f raw=(0x%04X,0x%04X,0x%04X)",
                preset_id,
                coord.pan,
                coord.tilt,
                coord.zoom,
                coord.raw_pan,
                coord.raw_tilt,
                coord.raw_zoom,
            )
        self.current_preset_idx = len(self.cfg.stage3.preset_ids) - 1
        self.last_known_ptz = self.preset_coords[self.cfg.stage3.preset_ids[self.current_preset_idx]]

    def _start_patrolling(self, preset_index: int, reason: str):
        preset_id = self.cfg.stage3.preset_ids[preset_index]
        self.ptz.stop()
        self._clear_tracking_context()
        self.next_preset_idx = preset_index
        self.ptz.goto_preset(preset_id)
        self.last_action = self.ptz.last_action
        self.settling_until = time.time() + self.cfg.stage3.patrol_settle_s
        self._mark_settle_pending(f"goto_preset_{preset_id}")
        self._set_state(Stage3State.PATROLLING, reason)
        logger.info("[TIMING] patrol_target preset=%d settle_s=%.1f", preset_id, self.cfg.stage3.patrol_settle_s)

    def _enter_observing(self):
        self.current_preset_idx = self.next_preset_idx
        preset_id = self._current_preset_id()
        self._clear_tracking_context()
        self.observe_until = time.time() + self.cfg.stage3.observe_hold_s
        self._set_state(Stage3State.OBSERVING, "patrol_arrived")
        logger.info("[TIMING] observe_start preset=%s hold_s=%.1f", preset_id, self.cfg.stage3.observe_hold_s)

    def _start_tracking(self, frame_age_ms: float):
        if self.state == Stage3State.TRACKING:
            return
        preset_id = self._current_preset_id()
        self.track_until = time.time() + self.cfg.stage3.tracking_timeout_s
        self.track_started_wall_ts = time.time()
        self.track_lost_since = 0.0
        self.face_zoom_ready_count = 0
        self._set_state(Stage3State.TRACKING, "face_found")
        self._mark_engagement_start(frame_age_ms)
        logger.info(
            "[TIMING] tracking_start preset=%s track_s=%.1f frame_age_ms=%.1f",
            preset_id,
            self.cfg.stage3.tracking_timeout_s,
            frame_age_ms,
        )

    def _recent_face(self, now: float) -> bool:
        return (now - self.last_face_seen_ts) <= self.cfg.stage3.lost_face_to_body_s

    def _recent_body(self, now: float) -> bool:
        return (now - self.last_body_seen_ts) <= self.cfg.stage3.lost_all_reset_s

    def _body_anchor_point(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = y1 + (y2 - y1) * self.cfg.stage3.body_anchor_ratio
        return cx, cy

    def _target_point(self, kind: str, bbox):
        if kind == "face":
            return bbox_center(bbox)
        return self._body_anchor_point(bbox)

    def _append_observation(self, kind: str, bbox, ts: float):
        x, y = self._target_point(kind, bbox)
        obs = TargetObservation(ts=ts, x=x, y=y, bbox=bbox)
        if kind == "face":
            self.face_history.append(obs)
        else:
            self.body_history.append(obs)

    def _predict_point(self, kind: str, bbox):
        history = self.face_history if kind == "face" else self.body_history
        lead_s = self.cfg.stage3.face_predict_lead_s if kind == "face" else self.cfg.stage3.body_predict_lead_s
        x, y = self._target_point(kind, bbox)
        if len(history) < 2:
            return x, y

        oldest = history[0]
        newest = history[-1]
        dt = newest.ts - oldest.ts
        if dt <= 1e-6:
            return x, y
        vx = (newest.x - oldest.x) / dt
        vy = (newest.y - oldest.y) / dt
        return newest.x + vx * lead_s, newest.y + vy * lead_s

    def _target_error(self, kind: str, bbox, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        px, py = self._predict_point(kind, bbox)
        target_y_ratio = self.cfg.stage3.face_target_y_ratio if kind == "face" else self.cfg.stage3.body_target_y_ratio
        dx = (px - frame_w / 2.0) / max(frame_w, 1)
        dy = (py - frame_h * target_y_ratio) / max(frame_h, 1)
        width_ratio = (bbox[2] - bbox[0]) / max(frame_w, 1)
        return dx, dy, width_ratio

    def _upward_limit_reached(self) -> bool:
        if self.ptz_scope is None:
            return False
        preset_id = self._current_preset_id()
        if preset_id is None or preset_id not in self.preset_coords:
            return False
        current, _ = get_ptz_coord(self.client, self.cfg.hik.channel)
        self.last_known_ptz = current
        preset_coord = self.preset_coords[preset_id]
        return current.tilt <= (preset_coord.tilt - self.cfg.stage3.max_upward_tilt_offset_deg)

    def _current_tolerances(self, target_kind: str):
        if target_kind == "face":
            return (
                self.cfg.stage3.face_center_tolerance_ratio_x,
                self.cfg.stage3.face_center_tolerance_ratio_y,
            )
        return (
            self.cfg.stage3.person_center_tolerance_ratio_x,
            self.cfg.stage3.person_center_tolerance_ratio_y,
        )

    def _adaptive_move_params(self, err_abs: float, tol: float):
        clamped = min(max(err_abs - tol, 0.0), 0.35)
        span = max(0.35 - tol, 1e-6)
        ratio = clamped / span
        speed_span = max(self.cfg.stage3.move_speed - self.cfg.stage3.move_speed_min, 0)
        speed = self.cfg.stage3.move_speed_min + round(speed_span * ratio)
        pulse_span = max(self.cfg.stage3.move_pulse_s - self.cfg.stage3.move_pulse_min_s, 0.0)
        pulse_s = self.cfg.stage3.move_pulse_min_s + pulse_span * ratio
        return max(speed, self.cfg.stage3.move_speed_min), max(pulse_s, self.cfg.stage3.move_pulse_min_s)

    def _nudge_towards(self, dx: float, dy: float, target_kind: str):
        command_name = None
        tol_x, tol_y = self._current_tolerances(target_kind)
        now_wall = time.time()

        if abs(dx) >= abs(dy) and abs(dx) > tol_x:
            speed, pulse_s = self._adaptive_move_params(abs(dx), tol_x)
            if dx < 0:
                self.ptz.pan_left_async(pulse_s, speed)
                command_name = "pan_left"
            else:
                self.ptz.pan_right_async(pulse_s, speed)
                command_name = "pan_right"
        elif abs(dy) > tol_y:
            speed, pulse_s = self._adaptive_move_params(abs(dy), tol_y)
            if dy < 0:
                if now_wall < self.tilt_up_blocked_until:
                    if abs(dx) > tol_x:
                        speed, pulse_s = self._adaptive_move_params(abs(dx), tol_x)
                        if dx < 0:
                            self.ptz.pan_left_async(pulse_s, speed)
                            command_name = "pan_left"
                        else:
                            self.ptz.pan_right_async(pulse_s, speed)
                            command_name = "pan_right"
                    else:
                        self.ptz.stop()
                        self.last_action = "tilt_up_blocked(cooldown)"
                        self.settling_until = now_wall
                        return
                elif self._upward_limit_reached():
                    self.tilt_up_blocked_until = now_wall + self.cfg.stage3.tilt_up_block_cooldown_s
                    self.ptz.stop()
                    self.last_action = "tilt_up_blocked(limit)"
                    self.settling_until = now_wall
                    logger.info(
                        "[TIMING] tilt_up_blocked preset=%s limit_deg=%.1f cooldown_s=%.1f",
                        self._current_preset_id(),
                        self.cfg.stage3.max_upward_tilt_offset_deg,
                        self.cfg.stage3.tilt_up_block_cooldown_s,
                    )
                    if abs(dx) > tol_x:
                        speed, pulse_s = self._adaptive_move_params(abs(dx), tol_x)
                        if dx < 0:
                            self.ptz.pan_left_async(pulse_s, speed)
                            command_name = "pan_left"
                        else:
                            self.ptz.pan_right_async(pulse_s, speed)
                            command_name = "pan_right"
                    else:
                        return
                else:
                    self.tilt_up_blocked_until = 0.0
                    self.ptz.tilt_up_async(pulse_s, speed)
                    command_name = "tilt_up"
            else:
                self.tilt_up_blocked_until = 0.0
                self.ptz.tilt_down_async(pulse_s, speed)
                command_name = "tilt_down"
        else:
            self.ptz.stop()

        self.last_action = self.ptz.last_action
        if command_name is not None:
            self.settling_until = time.time() + self.cfg.stage3.move_control_interval_s
            self._log_command_issue(command_name)
        else:
            self.settling_until = time.time()

    def _zoom_step(self):
        if self.zoom_steps >= self.cfg.stage3.max_zoom_steps:
            return
        self.ptz.stop()
        self.ptz.zoom_in(self.cfg.stage3.zoom_pulse_s, self.cfg.stage3.zoom_speed)
        self.zoom_steps += 1
        self.last_zoom_ts = time.time()
        self.last_action = self.ptz.last_action
        self.settling_until = time.time() + self.cfg.stage3.settle_after_zoom_s
        self._mark_settle_pending("zoom_in")
        self._log_command_issue("zoom_in")

    def _face_zoom_complete(self, width_ratio: float) -> bool:
        return width_ratio >= self.cfg.stage3.desired_face_ratio_min

    def _zoom_out_before_reset(self):
        if self.ptz_scope is None:
            return 0, None, None
        before, _ = get_ptz_coord(self.client, self.cfg.hik.channel)
        steps = 0
        for _ in range(self.cfg.stage3.reset_zoom_max_steps):
            current, _ = get_ptz_coord(self.client, self.cfg.hik.channel)
            self.last_known_ptz = current
            if current.zoom <= self.ptz_scope.zoom_min + 0.05:
                return steps, before, current
            self.ptz.zoom_out(self.cfg.stage3.zoom_pulse_s, self.cfg.stage3.zoom_speed)
            steps += 1
            time.sleep(self.cfg.stage3.reset_zoom_settle_s)
        final, _ = get_ptz_coord(self.client, self.cfg.hik.channel)
        self.last_known_ptz = final
        return steps, before, final

    def _begin_resetting(self, reason: str):
        self.ptz.stop()
        try:
            zoom_steps, zoom_before, zoom_after = self._zoom_out_before_reset()
            if zoom_before is not None and zoom_after is not None:
                logger.info(
                    "[TIMING] reset_zoom_out steps=%d zoom_before=%.1f zoom_after=%.1f",
                    zoom_steps,
                    zoom_before.zoom,
                    zoom_after.zoom,
                )
        except Exception as exc:
            logger.warning("reset zoom-out failed: %s", exc)

        current, _ = get_ptz_coord(self.client, self.cfg.hik.channel)
        self.last_known_ptz = current
        preset_id, score = nearest_preset(current, self.preset_coords, self.ptz_scope)
        preset_index = self.cfg.stage3.preset_ids.index(preset_id)

        logger.info(
            "[TIMING] reset_to_preset reason=%s nearest_preset=%d score=%.6f pan=%.1f tilt=%.1f zoom=%.1f",
            reason,
            preset_id,
            score,
            current.pan,
            current.tilt,
            current.zoom,
        )

        self._reset_engagement(reason)
        self._clear_tracking_context()
        self.next_preset_idx = preset_index
        self.ptz.goto_preset(preset_id)
        self.last_action = self.ptz.last_action
        self.settling_until = time.time() + self.cfg.stage3.reset_settle_s
        self._mark_settle_pending(f"reset_preset_{preset_id}")
        self._set_state(Stage3State.RESETTING, reason)

    def _tracking_step(self, tracked_person, tracked_face, frame_shape, now_wall: float):
        target_face_valid = tracked_face is not None and self._recent_face(now_wall)
        target_body_valid = tracked_person is not None and self._recent_body(now_wall)

        if not target_face_valid and not target_body_valid:
            if self.track_lost_since <= 0.0:
                self.track_lost_since = now_wall
                logger.info(
                    "[TIMING] tracking_reacquire_start grace_s=%.1f",
                    self.cfg.stage3.tracking_lost_grace_s,
                )
            grace_until = self.track_lost_since + self.cfg.stage3.tracking_lost_grace_s
            if now_wall < max(self.settling_until, grace_until):
                self.ptz.stop()
                self.last_action = "tracking_reacquire_wait"
                return "none", None, 0.0, 0.0, 0.0
            self._begin_resetting("lost_target")
            return "none", None, 0.0, 0.0, 0.0
        self.track_lost_since = 0.0

        if target_face_valid:
            target_kind = "face"
            target_bbox = tracked_face.bbox
        else:
            target_kind = "body"
            target_bbox = tracked_person.bbox
            self.face_zoom_ready_count = 0

        dx, dy, width_ratio = self._target_error(target_kind, target_bbox, frame_shape)
        self._current_timing["target_kind"] = target_kind

        if now_wall < self.settling_until:
            return target_kind, target_bbox, dx, dy, width_ratio

        if target_kind == "face":
            centered = (
                abs(dx) <= self.cfg.stage3.face_zoom_ready_ratio_x
                and abs(dy) <= self.cfg.stage3.face_zoom_ready_ratio_y
            )
            if centered:
                self.face_zoom_ready_count += 1
            else:
                self.face_zoom_ready_count = 0

            if centered and not self._face_zoom_complete(width_ratio):
                can_zoom = (
                    self.zoom_steps < self.cfg.stage3.max_zoom_steps
                    and self.face_zoom_ready_count >= self.cfg.stage3.zoom_ready_consecutive_frames
                    and (time.time() - self.last_zoom_ts) >= self.cfg.stage3.min_zoom_interval_s
                    and (now_wall - self.track_started_wall_ts) >= self.cfg.stage3.zoom_cooldown_after_tracking_start_s
                )
                if can_zoom:
                    logger.info(
                        "[TIMING] zoom_ready target=%s dx=%.3f dy=%.3f ratio=%.3f ready_frames=%d",
                        target_kind,
                        dx,
                        dy,
                        width_ratio,
                        self.face_zoom_ready_count,
                    )
                    self._zoom_step()
                    self.face_zoom_ready_count = 0
                else:
                    self.ptz.stop()
                    self.last_action = self.ptz.last_action
            elif abs(dx) > self.cfg.stage3.face_center_tolerance_ratio_x or abs(dy) > self.cfg.stage3.face_center_tolerance_ratio_y:
                self._nudge_towards(dx, dy, "face")
            else:
                self.ptz.stop()
                self.last_action = self.ptz.last_action
        else:
            if abs(dx) > self.cfg.stage3.person_center_tolerance_ratio_x or abs(dy) > self.cfg.stage3.person_center_tolerance_ratio_y:
                self._nudge_towards(dx, dy, "body")
            else:
                self.ptz.stop()
                self.last_action = self.ptz.last_action

        return target_kind, target_bbox, dx, dy, width_ratio

    def run(self):
        setup_logger("app", self.cfg.log.level, self.cfg.log.file)
        providers = auto_providers(self.cfg.runtime.prefer_gpu)

        self.video = RtspVideoSource(self.cfg.hik.rtsp_url)
        if self.enable_web:
            self.web = WebDebugServer(
                host=self.cfg.web.host,
                port=self.cfg.web.port,
                jpeg_quality=self.cfg.web.jpeg_quality,
            )
        else:
            logger.info("Web debug server disabled for timing test mode")
            self.web = None

        self.person_detector = PersonDetector(
            self.cfg.models.person,
            input_size=640,
            conf=self.cfg.detect.person_conf,
            iou=self.cfg.detect.person_iou,
            providers=providers,
            edge_reject_enabled=self.cfg.detect.edge_reject_enabled,
            edge_margin=self.cfg.detect.edge_margin,
        )
        self.face_detector = FaceDetector(
            self.cfg.models.face,
            input_size=640,
            conf=self.cfg.detect.face_conf,
            iou=self.cfg.detect.face_iou,
            providers=providers,
            edge_reject_enabled=self.cfg.detect.edge_reject_enabled,
            edge_margin=self.cfg.detect.edge_margin,
        )

        self.client = HikClient(
            lib_dir=self.cfg.hik.sdk_lib_dir,
            ip=self.cfg.hik.ip,
            port=self.cfg.hik.port,
            user=self.cfg.hik.user,
            password=self.cfg.hik.password,
            channel=self.cfg.hik.channel,
        )
        self.ptz = HikManualPtzController(
            self.client,
            default_move_speed=self.cfg.stage3.move_speed,
            default_zoom_speed=self.cfg.stage3.zoom_speed,
        )

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("Stage3 starting")
        self.ptz.login()
        self._initialize_preset_map()
        self._start_patrolling(0, "startup_patrol")
        self.video.start()
        if self.web is not None:
            self.web.start()
        self._wait_first_frame()

        try:
            while not self.stop_flag:
                loop_perf_start = time.perf_counter()
                frame = self.video.read()
                if frame is None:
                    time.sleep(self.cfg.stage3.loop_sleep_s)
                    continue

                now_wall = time.time()
                self._update_fps()
                self.ptz.tick()
                self._maybe_log_settle_done()

                frame_age_ms = 0.0
                if self.video.timestamp > 0:
                    frame_age_ms = max(0.0, (now_wall - self.video.timestamp) * 1000)

                person_dets = []
                face_dets = []
                tracked_person = None
                tracked_face = None
                target_kind = "none"
                target_bbox = None
                dx = 0.0
                dy = 0.0
                target_ratio = 0.0
                person_det_ms = 0.0
                face_det_ms = 0.0

                self._current_timing = {
                    "loop_perf_start": loop_perf_start,
                    "person_det_ms": 0.0,
                    "face_det_ms": 0.0,
                    "frame_age_ms": frame_age_ms,
                    "target_kind": "none",
                }

                if self.paused:
                    self.ptz.stop()
                    self.last_action = "paused"
                elif self.state == Stage3State.PATROLLING:
                    if now_wall >= self.settling_until:
                        self._enter_observing()
                    else:
                        self.last_action = f"patrolling_to_preset({self.cfg.stage3.preset_ids[self.next_preset_idx]})"
                elif self.state == Stage3State.RESETTING:
                    if now_wall >= self.settling_until:
                        self._enter_observing()
                    else:
                        self.last_action = f"resetting_to_preset({self.cfg.stage3.preset_ids[self.next_preset_idx]})"
                else:
                    person_det_started = time.perf_counter()
                    person_dets = self.person_detector.detect(frame)
                    person_det_ms = (time.perf_counter() - person_det_started) * 1000
                    tracked_person = self.person_tracker.update(person_dets, frame.shape, now=now_wall)
                    if person_dets:
                        self.last_body_seen_ts = now_wall
                        self._append_observation("body", tracked_person.bbox, now_wall)

                    face_det_started = time.perf_counter()
                    face_dets = self.face_detector.detect(frame)
                    face_det_ms = (time.perf_counter() - face_det_started) * 1000
                    tracked_face = self.face_tracker.update(face_dets, frame.shape, now=now_wall)
                    if face_dets:
                        self.last_face_seen_ts = now_wall
                        self._append_observation("face", tracked_face.bbox, now_wall)

                    self._current_timing.update(
                        {
                            "person_det_ms": person_det_ms,
                            "face_det_ms": face_det_ms,
                        }
                    )

                    if self.state == Stage3State.OBSERVING:
                        if tracked_face is not None and face_dets:
                            self._start_tracking(frame_age_ms)
                            target_kind, target_bbox, dx, dy, target_ratio = self._tracking_step(
                                tracked_person, tracked_face, frame.shape, now_wall
                            )
                        elif now_wall >= self.observe_until:
                            self._start_patrolling(self._next_index(), "observe_timeout")
                        else:
                            self.last_action = "observing_wait_face"
                    elif self.state == Stage3State.TRACKING:
                        if now_wall >= self.track_until:
                            self._begin_resetting("follow_timeout")
                        else:
                            target_kind, target_bbox, dx, dy, target_ratio = self._tracking_step(
                                tracked_person, tracked_face, frame.shape, now_wall
                            )

                debug = {
                    "dx": f"{dx:+.3f}",
                    "dy": f"{dy:+.3f}",
                    "target_ratio": f"{target_ratio:.3f}",
                    "zoom_steps": self.zoom_steps,
                    "last_action": self.last_action,
                    "paused": self.paused,
                    "settling": now_wall < self.settling_until,
                    "fps": f"{self.fps:.1f}",
                    "preset": self._current_preset_id() if self._current_preset_id() is not None else "-",
                }
                if self.web is not None:
                    annotated = self.overlay.render(
                        frame,
                        state=self.state.value,
                        person_dets=person_dets,
                        face_dets=face_dets,
                        tracked_person=tracked_person,
                        target_kind=target_kind,
                        target_bbox=target_bbox,
                        debug=debug,
                        draw_all=self.cfg.stage3.debug_draw_all,
                    )
                    self.web.show(annotated)

                loop_ms = (time.perf_counter() - loop_perf_start) * 1000
                self._record_timing_summary(
                    person_det_ms=person_det_ms,
                    face_det_ms=face_det_ms,
                    loop_ms=loop_ms,
                    frame_age_ms=frame_age_ms,
                )
                time.sleep(self.cfg.stage3.loop_sleep_s)
        finally:
            logger.info("Stage3 shutting down")
            try:
                self.video.stop()
            except Exception:
                pass
            try:
                self.ptz.stop()
            except Exception:
                pass
            try:
                self.ptz.logout()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Stage3 preset patrol with simple tracking")
    parser.add_argument("--common-config", default="config/common.yaml")
    parser.add_argument("--stage-config", default="config/stage3_preset_patrol.yaml")
    parser.add_argument("--no-web", action="store_true", help="Disable MJPEG web streaming")
    parser.add_argument("--timing-log-interval", type=float, default=1.0, help="Seconds between timing summary logs")
    args = parser.parse_args()

    app = Stage3PresetPatrolApp(
        args.common_config,
        args.stage_config,
        enable_web=not args.no_web,
        timing_log_interval_s=args.timing_log_interval,
    )
    app.run()


if __name__ == "__main__":
    main()
