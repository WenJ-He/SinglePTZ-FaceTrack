"""Stage 2: single-person moving follow with face-priority and body fallback."""

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
from src.stage1_single_static import Stage1SingleStaticApp
from src.tracker import SingleTargetTracker
from src.video_source import RtspVideoSource
from src.web import Stage1Overlay, WebDebugServer

logger = logging.getLogger("app")


class Stage2State(str, Enum):
    SEARCH_WIDE = "SEARCH_WIDE"
    ACQUIRE_TARGET = "ACQUIRE_TARGET"
    ZOOM_TO_MAX = "ZOOM_TO_MAX"
    FOLLOW_FACE = "FOLLOW_FACE"
    FOLLOW_BODY = "FOLLOW_BODY"
    LOST_TARGET = "LOST_TARGET"
    RESTORE_HOME = "RESTORE_HOME"


@dataclass
class TargetObservation:
    ts: float
    x: float
    y: float
    bbox: tuple[int, int, int, int]


class Stage2SingleMovingApp(Stage1SingleStaticApp):
    """Stage-2 runtime for a single moving person."""

    def __init__(self, common_config: str, stage_config: str, enable_web: bool = True, timing_log_interval_s: float = 1.0):
        super().__init__(common_config, stage_config, enable_web=enable_web, timing_log_interval_s=timing_log_interval_s)
        self.state = Stage2State.SEARCH_WIDE
        self.person_tracker = SingleTargetTracker(lost_timeout_s=self.cfg.stage2.lost_timeout_s)
        self.face_tracker = SingleTargetTracker(lost_timeout_s=self.cfg.stage2.lost_face_to_body_s)
        self.overlay = Stage1Overlay()
        self.face_history = deque(maxlen=6)
        self.body_history = deque(maxlen=6)
        self.last_face_seen_ts = 0.0
        self.last_body_seen_ts = 0.0
        self.follow_started_at = 0.0
        self.follow_until = 0.0

    def _set_state(self, new_state: Stage2State, reason: str = ""):
        if self.state != new_state:
            suffix = f" ({reason})" if reason else ""
            logger.info("STATE %s -> %s%s", self.state.value, new_state.value, suffix)
            self.state = new_state
        else:
            self.state = new_state

    def _clear_motion_context(self):
        self.person_tracker.reset()
        self.face_tracker.reset()
        self.face_history.clear()
        self.body_history.clear()
        self.last_face_seen_ts = 0.0
        self.last_body_seen_ts = 0.0
        self.follow_started_at = 0.0
        self.follow_until = 0.0
        self.zoom_steps = 0
        self.last_zoom_ts = 0.0
        self.settling_until = 0.0

    def _restore_home(self, reason: str):
        logger.info("[TIMING] restore_home reason=%s zoom_steps=%d", reason, self.zoom_steps)
        self._clear_motion_context()
        self.ptz.stop()
        self.ptz.goto_preset(self.cfg.stage2.home_preset)
        self.last_action = self.ptz.last_action
        self.settling_until = time.time() + self.cfg.stage2.restore_home_settle_s
        self._mark_settle_pending("restore_home")
        self._reset_engagement(reason)
        self._set_state(Stage2State.RESTORE_HOME, "restore_home")

    def _handle_command(self, cmd: str):
        cmd = cmd.strip().lower()
        if not cmd:
            return
        if cmd in {"q", "quit"}:
            self.stop_flag = True
            return
        if cmd in {"p", "pause"}:
            self.paused = not self.paused
            return
        if cmd in {"h", "home"}:
            self._restore_home("manual_home")
            return

        manual_map = {
            "left": lambda: self.ptz.pan_left(self.cfg.stage2.move_pulse_s, self.cfg.stage2.move_speed),
            "right": lambda: self.ptz.pan_right(self.cfg.stage2.move_pulse_s, self.cfg.stage2.move_speed),
            "up": lambda: self.ptz.tilt_up(self.cfg.stage2.move_pulse_s, self.cfg.stage2.move_speed),
            "down": lambda: self.ptz.tilt_down(self.cfg.stage2.move_pulse_s, self.cfg.stage2.move_speed),
            "zoom_in": lambda: self.ptz.zoom_in(self.cfg.stage2.zoom_pulse_s, self.cfg.stage2.zoom_speed),
            "zoom_out": lambda: self.ptz.zoom_out(self.cfg.stage2.zoom_pulse_s, self.cfg.stage2.zoom_speed),
            "stop": self.ptz.stop,
        }
        handler = manual_map.get(cmd)
        if handler is None:
            return
        handler()
        self.last_action = self.ptz.last_action
        if cmd in {"zoom_in", "zoom_out"}:
            self.settling_until = time.time() + self.cfg.stage2.settle_after_zoom_s
            self._mark_settle_pending(cmd)
        else:
            self.settling_until = time.time() + self.cfg.stage2.settle_after_move_s
            self._mark_settle_pending(cmd)

    def _current_tolerances(self, target_kind: str):
        if target_kind == "face":
            return (
                self.cfg.stage2.face_center_tolerance_ratio_x,
                self.cfg.stage2.face_center_tolerance_ratio_y,
            )
        return (
            self.cfg.stage2.person_center_tolerance_ratio_x,
            self.cfg.stage2.person_center_tolerance_ratio_y,
        )

    def _adaptive_move_params(self, err_abs: float, tol: float):
        clamped = min(max(err_abs - tol, 0.0), 0.35)
        span = max(0.35 - tol, 1e-6)
        ratio = clamped / span
        speed_span = max(self.cfg.stage2.move_speed - self.cfg.stage2.move_speed_min, 0)
        speed = self.cfg.stage2.move_speed_min + round(speed_span * ratio)
        pulse_span = max(self.cfg.stage2.move_pulse_s - self.cfg.stage2.move_pulse_min_s, 0.0)
        pulse_s = self.cfg.stage2.move_pulse_min_s + pulse_span * ratio
        return max(speed, self.cfg.stage2.move_speed_min), max(pulse_s, self.cfg.stage2.move_pulse_min_s)

    def _nudge_towards(self, dx: float, dy: float, target_kind: str):
        command_name = None
        tol_x, tol_y = self._current_tolerances(target_kind)

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
                self.ptz.tilt_up_async(pulse_s, speed)
                command_name = "tilt_up"
            else:
                self.ptz.tilt_down_async(pulse_s, speed)
                command_name = "tilt_down"
        else:
            self.ptz.stop()

        self.last_action = self.ptz.last_action
        if command_name is not None:
            self.settling_until = time.time() + self.cfg.stage2.move_control_interval_s
            self._log_command_issue(command_name)
        else:
            self.settling_until = time.time()

    def _zoom_step(self):
        if self.zoom_steps >= self.cfg.stage2.max_zoom_steps:
            return
        self.ptz.stop()
        self.ptz.zoom_in(self.cfg.stage2.zoom_pulse_s, self.cfg.stage2.zoom_speed)
        self.zoom_steps += 1
        self.last_zoom_ts = time.time()
        self.last_action = self.ptz.last_action
        self._set_state(Stage2State.ZOOM_TO_MAX, "auto_zoom")
        self.settling_until = time.time() + self.cfg.stage2.settle_after_zoom_s
        self._mark_settle_pending("zoom_in")
        self._log_command_issue("zoom_in")

    def _body_anchor_point(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = y1 + (y2 - y1) * self.cfg.stage2.body_anchor_ratio
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

    def _predict_point(self, kind: str, bbox, now: float):
        history = self.face_history if kind == "face" else self.body_history
        lead_s = self.cfg.stage2.face_predict_lead_s if kind == "face" else self.cfg.stage2.body_predict_lead_s
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

    def _target_error(self, kind: str, bbox, frame_shape, now: float):
        frame_h, frame_w = frame_shape[:2]
        px, py = self._predict_point(kind, bbox, now)
        dx = (px - frame_w / 2.0) / max(frame_w, 1)
        dy = (py - frame_h / 2.0) / max(frame_h, 1)
        width_ratio = (bbox[2] - bbox[0]) / max(frame_w, 1)
        return dx, dy, width_ratio

    def _follow_active(self):
        return self.follow_until > 0.0 and time.time() < self.follow_until

    def _begin_follow(self, kind: str):
        if self.follow_started_at <= 0.0:
            self.follow_started_at = time.perf_counter()
            self.follow_until = time.time() + self.cfg.stage2.follow_duration_s
            logger.info("[TIMING] follow_start kind=%s follow_s=%.1f zoom_steps=%d", kind, self.cfg.stage2.follow_duration_s, self.zoom_steps)
        self._set_state(Stage2State.FOLLOW_FACE if kind == "face" else Stage2State.FOLLOW_BODY, "follow_begin")

    def _follow_timed_out(self):
        return self.follow_until > 0.0 and time.time() >= self.follow_until

    def _recent_face(self, now: float) -> bool:
        return (now - self.last_face_seen_ts) <= self.cfg.stage2.lost_face_to_body_s

    def _recent_body(self, now: float) -> bool:
        return (now - self.last_body_seen_ts) <= self.cfg.stage2.lost_all_restore_s

    def _face_zoom_complete(self, width_ratio: float) -> bool:
        return width_ratio >= self.cfg.stage2.desired_face_ratio_min

    def _control_acquire(self, target_kind: str, target_bbox, frame_shape, now: float):
        dx, dy, width_ratio = self._target_error(target_kind, target_bbox, frame_shape, now)
        self._current_timing["target_kind"] = target_kind

        if target_kind == "face":
            ready = (
                abs(dx) <= self.cfg.stage2.face_zoom_ready_ratio_x
                and abs(dy) <= self.cfg.stage2.face_zoom_ready_ratio_y
            )
        else:
            ready = (
                abs(dx) <= self.cfg.stage2.person_center_tolerance_ratio_x
                and abs(dy) <= self.cfg.stage2.person_center_tolerance_ratio_y
            )

        if ready:
            self.ptz.stop()
            if target_kind == "face" and self._face_zoom_complete(width_ratio):
                logger.info(
                    "[TIMING] follow_ready target=%s dx=%.3f dy=%.3f ratio=%.3f zoom_steps=%d",
                    target_kind,
                    dx,
                    dy,
                    width_ratio,
                    self.zoom_steps,
                )
                self._begin_follow("face")
            elif self.zoom_steps < self.cfg.stage2.max_zoom_steps:
                self._set_state(Stage2State.ZOOM_TO_MAX, "target_ready")
            else:
                self._begin_follow("face" if target_kind == "face" else "body")
        else:
            self._set_state(Stage2State.ACQUIRE_TARGET, f"{target_kind}_offset")
            self._nudge_towards(dx, dy, target_kind)
        return dx, dy, width_ratio

    def _control_zoom(self, target_kind: str, target_bbox, frame_shape, now: float):
        dx, dy, width_ratio = self._target_error(target_kind, target_bbox, frame_shape, now)
        self._current_timing["target_kind"] = target_kind

        if target_kind == "face":
            zoom_ready = (
                abs(dx) <= self.cfg.stage2.face_zoom_ready_ratio_x
                and abs(dy) <= self.cfg.stage2.face_zoom_ready_ratio_y
            )
        else:
            zoom_ready = (
                abs(dx) <= self.cfg.stage2.person_center_tolerance_ratio_x
                and abs(dy) <= self.cfg.stage2.person_center_tolerance_ratio_y
            )

        if not zoom_ready:
            self._set_state(Stage2State.ACQUIRE_TARGET, f"{target_kind}_offset")
            self._nudge_towards(dx, dy, target_kind)
        elif target_kind == "face" and self._face_zoom_complete(width_ratio):
            self.ptz.stop()
            logger.info(
                "[TIMING] follow_ready target=%s dx=%.3f dy=%.3f ratio=%.3f zoom_steps=%d",
                target_kind,
                dx,
                dy,
                width_ratio,
                self.zoom_steps,
            )
            self._begin_follow("face")
        elif self.zoom_steps < self.cfg.stage2.max_zoom_steps and (time.time() - self.last_zoom_ts) >= self.cfg.stage2.min_zoom_interval_s:
            self.ptz.stop()
            logger.info(
                "[TIMING] zoom_ready target=%s dx=%.3f dy=%.3f ratio=%.3f",
                target_kind,
                dx,
                dy,
                width_ratio,
            )
            self._zoom_step()
        elif self.zoom_steps >= self.cfg.stage2.max_zoom_steps:
            self._begin_follow("face" if target_kind == "face" else "body")
        return dx, dy, width_ratio

    def _control_follow(self, target_kind: str, target_bbox, frame_shape, now: float):
        dx, dy, width_ratio = self._target_error(target_kind, target_bbox, frame_shape, now)
        self._current_timing["target_kind"] = target_kind
        if target_kind == "face":
            self._set_state(Stage2State.FOLLOW_FACE, "follow_face")
            if abs(dx) > self.cfg.stage2.face_center_tolerance_ratio_x or abs(dy) > self.cfg.stage2.face_center_tolerance_ratio_y:
                self._nudge_towards(dx, dy, "face")
        else:
            self._set_state(Stage2State.FOLLOW_BODY, "follow_body")
            if abs(dx) > self.cfg.stage2.person_center_tolerance_ratio_x or abs(dy) > self.cfg.stage2.person_center_tolerance_ratio_y:
                self._nudge_towards(dx, dy, "body")
        return dx, dy, width_ratio

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

        client = HikClient(
            lib_dir=self.cfg.hik.sdk_lib_dir,
            ip=self.cfg.hik.ip,
            port=self.cfg.hik.port,
            user=self.cfg.hik.user,
            password=self.cfg.hik.password,
            channel=self.cfg.hik.channel,
        )
        self.ptz = HikManualPtzController(
            client,
            default_move_speed=self.cfg.stage2.move_speed,
            default_zoom_speed=self.cfg.stage2.zoom_speed,
        )

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("Stage2 starting")
        self.ptz.login()
        self._restore_home("startup")
        self.video.start()
        if self.web is not None:
            self.web.start()
        self._wait_first_frame()

        try:
            while not self.stop_flag:
                loop_perf_start = time.perf_counter()
                frame = self.video.read()
                if frame is None:
                    time.sleep(self.cfg.stage2.loop_sleep_s)
                    continue

                now_wall = time.time()
                self._update_fps()
                self.ptz.tick()
                self._maybe_log_settle_done()

                if self.state == Stage2State.RESTORE_HOME:
                    if now_wall >= self.settling_until:
                        logger.info("[TIMING] restore_home_done home_preset=%d", self.cfg.stage2.home_preset)
                        self._set_state(Stage2State.SEARCH_WIDE, "restore_done")
                    time.sleep(self.cfg.stage2.loop_sleep_s)
                    continue

                frame_age_ms = 0.0
                if self.video.timestamp > 0:
                    frame_age_ms = max(0.0, (now_wall - self.video.timestamp) * 1000)

                cmd = self.web.poll_command() if self.web is not None else None
                if cmd:
                    self._handle_command(cmd)

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

                target_kind = "none"
                target_bbox = None
                dx = 0.0
                dy = 0.0
                target_ratio = 0.0
                self._current_timing = {
                    "loop_perf_start": loop_perf_start,
                    "person_det_ms": person_det_ms,
                    "face_det_ms": face_det_ms,
                    "frame_age_ms": frame_age_ms,
                    "target_kind": "none",
                }

                if self.paused:
                    self.ptz.stop()
                    pass
                elif self._follow_timed_out():
                    self._restore_home("follow_timeout")
                    time.sleep(self.cfg.stage2.loop_sleep_s)
                    continue
                elif self.state == Stage2State.SEARCH_WIDE:
                    if tracked_face is not None and self._recent_face(now_wall):
                        self._mark_engagement_start(frame_age_ms)
                        target_kind = "face"
                        target_bbox = tracked_face.bbox
                        self._set_state(Stage2State.ACQUIRE_TARGET, "face_found")
                        dx, dy, target_ratio = self._control_acquire(target_kind, target_bbox, frame.shape, now_wall)
                    elif tracked_person is not None and self._recent_body(now_wall):
                        self._mark_engagement_start(frame_age_ms)
                        target_kind = "body"
                        target_bbox = tracked_person.bbox
                        self._set_state(Stage2State.ACQUIRE_TARGET, "body_found")
                        dx, dy, target_ratio = self._control_acquire(target_kind, target_bbox, frame.shape, now_wall)
                    else:
                        self.ptz.stop()
                        self.last_action = "waiting_for_person"
                else:
                    target_face_valid = tracked_face is not None and self._recent_face(now_wall)
                    target_body_valid = tracked_person is not None and self._recent_body(now_wall)

                    if not target_face_valid and not target_body_valid:
                        if now_wall < self.settling_until:
                            self.last_action = "settling_wait_target"
                        else:
                            self._set_state(Stage2State.LOST_TARGET, "no_target")
                            self._restore_home("lost_target")
                            time.sleep(self.cfg.stage2.loop_sleep_s)
                            continue
                    else:
                        self._mark_engagement_start(frame_age_ms)
                        if self.state in {Stage2State.ACQUIRE_TARGET, Stage2State.ZOOM_TO_MAX}:
                            if target_face_valid:
                                target_kind = "face"
                                target_bbox = tracked_face.bbox
                            elif target_body_valid:
                                target_kind = "body"
                                target_bbox = tracked_person.bbox

                            if now_wall >= self.settling_until:
                                if self.state == Stage2State.ACQUIRE_TARGET:
                                    dx, dy, target_ratio = self._control_acquire(target_kind, target_bbox, frame.shape, now_wall)
                                else:
                                    dx, dy, target_ratio = self._control_zoom(target_kind, target_bbox, frame.shape, now_wall)
                            else:
                                dx, dy, target_ratio = self._target_error(target_kind, target_bbox, frame.shape, now_wall)
                                self._current_timing["target_kind"] = target_kind
                        elif self.state == Stage2State.FOLLOW_FACE:
                            if target_face_valid:
                                target_kind = "face"
                                target_bbox = tracked_face.bbox
                                if now_wall >= self.settling_until:
                                    dx, dy, target_ratio = self._control_follow(target_kind, target_bbox, frame.shape, now_wall)
                                else:
                                    dx, dy, target_ratio = self._target_error(target_kind, target_bbox, frame.shape, now_wall)
                            elif target_body_valid:
                                self._set_state(Stage2State.FOLLOW_BODY, "face_lost_use_body")
                        elif self.state == Stage2State.FOLLOW_BODY:
                            if target_face_valid:
                                self._set_state(Stage2State.FOLLOW_FACE, "face_reacquired")
                            elif target_body_valid:
                                target_kind = "body"
                                target_bbox = tracked_person.bbox
                                if now_wall >= self.settling_until:
                                    dx, dy, target_ratio = self._control_follow(target_kind, target_bbox, frame.shape, now_wall)
                                else:
                                    dx, dy, target_ratio = self._target_error(target_kind, target_bbox, frame.shape, now_wall)
                        else:
                            if target_face_valid:
                                target_kind = "face"
                                target_bbox = tracked_face.bbox
                                if now_wall >= self.settling_until:
                                    dx, dy, target_ratio = self._control_follow(target_kind, target_bbox, frame.shape, now_wall)
                                else:
                                    dx, dy, target_ratio = self._target_error(target_kind, target_bbox, frame.shape, now_wall)
                            elif target_body_valid:
                                target_kind = "body"
                                target_bbox = tracked_person.bbox
                                if self.zoom_steps >= self.cfg.stage2.max_zoom_steps:
                                    if self.follow_started_at <= 0.0:
                                        self._begin_follow("body")
                                    if now_wall >= self.settling_until:
                                        dx, dy, target_ratio = self._control_follow(target_kind, target_bbox, frame.shape, now_wall)
                                    else:
                                        dx, dy, target_ratio = self._target_error(target_kind, target_bbox, frame.shape, now_wall)
                                else:
                                    if now_wall >= self.settling_until:
                                        dx, dy, target_ratio = self._control_zoom(target_kind, target_bbox, frame.shape, now_wall)
                                    else:
                                        dx, dy, target_ratio = self._target_error(target_kind, target_bbox, frame.shape, now_wall)

                        if self.state == Stage2State.ZOOM_TO_MAX and self.zoom_steps >= self.cfg.stage2.max_zoom_steps:
                            if target_face_valid:
                                self._begin_follow("face")
                            elif target_body_valid:
                                self._begin_follow("body")

                debug = {
                    "dx": f"{dx:+.3f}",
                    "dy": f"{dy:+.3f}",
                    "target_ratio": f"{target_ratio:.3f}",
                    "zoom_steps": self.zoom_steps,
                    "last_action": self.last_action,
                    "paused": self.paused,
                    "settling": now_wall < self.settling_until,
                    "fps": f"{self.fps:.1f}",
                }
                if self.web is not None:
                    annotated = self.overlay.render(
                        frame,
                        state=self.state.value,
                        person_dets=person_dets,
                        face_dets=face_dets,
                        tracked_person=tracked_person if tracked_person is not None else None,
                        target_kind=target_kind,
                        target_bbox=target_bbox,
                        debug=debug,
                        draw_all=self.cfg.stage2.debug_draw_all,
                    )
                    self.web.show(annotated)

                loop_ms = (time.perf_counter() - loop_perf_start) * 1000
                self._record_timing_summary(
                    person_det_ms=person_det_ms,
                    face_det_ms=face_det_ms,
                    loop_ms=loop_ms,
                    frame_age_ms=frame_age_ms,
                )
                time.sleep(self.cfg.stage2.loop_sleep_s)
        finally:
            logger.info("Stage2 shutting down")
            try:
                self.video.stop()
            except Exception:
                pass
            try:
                self.ptz.stop()
                self.ptz.goto_preset(self.cfg.stage2.home_preset)
                time.sleep(self.cfg.stage2.restore_home_settle_s)
            except Exception:
                pass
            try:
                self.ptz.logout()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Stage2 single-moving runtime")
    parser.add_argument("--common-config", default="config/common.yaml")
    parser.add_argument("--stage-config", default="config/stage2_single_moving.yaml")
    parser.add_argument("--no-web", action="store_true", help="Disable MJPEG web streaming to reduce debug overhead")
    parser.add_argument("--timing-log-interval", type=float, default=1.0, help="Seconds between timing summary logs")
    args = parser.parse_args()

    app = Stage2SingleMovingApp(
        args.common_config,
        args.stage_config,
        enable_web=not args.no_web,
        timing_log_interval_s=args.timing_log_interval,
    )
    app.run()


if __name__ == "__main__":
    main()
