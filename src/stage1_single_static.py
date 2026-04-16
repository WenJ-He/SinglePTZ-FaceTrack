"""Stage 1: single-person static closed-loop alignment."""

import argparse
import logging
import signal
import time
from enum import Enum

from src.app_config import auto_providers, load_app_config
from src.detector import FaceDetector, PersonDetector
from src.geometry import bbox_center
from src.hik_sdk import HikClient
from src.logger import setup_logger
from src.ptz_controller import HikManualPtzController
from src.target_selector import SingleTargetSelector
from src.tracker import SingleTargetTracker
from src.video_source import RtspVideoSource
from src.web import Stage1Overlay, WebDebugServer

logger = logging.getLogger("app")


class Stage1State(str, Enum):
    SEARCH_PERSON = "SEARCH_PERSON"
    CENTER_PERSON = "CENTER_PERSON"
    ZOOM_STEP = "ZOOM_STEP"
    CENTER_FACE = "CENTER_FACE"
    HOLD_FACE = "HOLD_FACE"
    RESTORE_HOME = "RESTORE_HOME"
    LOST_TARGET = "LOST_TARGET"


class Stage1SingleStaticApp:
    """Minimal stage-1 runtime for single-person static alignment."""

    def __init__(self, common_config: str, stage_config: str, enable_web: bool = True, timing_log_interval_s: float = 1.0):
        self.cfg = load_app_config(common_config, stage_config)
        self.enable_web = enable_web
        self.timing_log_interval_s = timing_log_interval_s
        self.stop_flag = False
        self.paused = False
        self.state = Stage1State.SEARCH_PERSON
        self.settling_until = 0.0
        self.last_zoom_ts = 0.0
        self.zoom_steps = 0
        self.last_action = "idle"
        self.fps = 0.0
        self._fps_ts = time.time()
        self._fps_count = 0
        self._pending_settle_reason = None
        self._settle_started_at = None
        self._engagement_started_at = None
        self._hold_logged = False
        self._hold_until = 0.0
        self._current_timing = {}
        self._timing_last_log_ts = time.time()
        self._timing_acc = {
            "count": 0,
            "person_det_ms": 0.0,
            "face_det_ms": 0.0,
            "loop_ms": 0.0,
            "frame_age_ms": 0.0,
        }

        self.person_tracker = SingleTargetTracker(lost_timeout_s=self.cfg.stage1.lost_timeout_s)
        self.overlay = Stage1Overlay()

        self.video = None
        self.web = None
        self.person_detector = None
        self.face_detector = None
        self.ptz = None

    def _set_state(self, new_state: Stage1State, reason: str = ""):
        if self.state != new_state:
            suffix = f" ({reason})" if reason else ""
            logger.info("STATE %s -> %s%s", self.state.value, new_state.value, suffix)
            self.state = new_state
        else:
            self.state = new_state

    def _mark_settle_pending(self, reason: str):
        self._pending_settle_reason = reason
        self._settle_started_at = time.perf_counter()

    def _maybe_log_settle_done(self):
        if self._pending_settle_reason is None:
            return
        if time.time() < self.settling_until:
            return
        settle_ms = 0.0
        if self._settle_started_at is not None:
            settle_ms = (time.perf_counter() - self._settle_started_at) * 1000
        logger.info(
            "[TIMING] settle_done reason=%s settle_ms=%.1f state=%s",
            self._pending_settle_reason,
            settle_ms,
            self.state.value,
        )
        self._pending_settle_reason = None
        self._settle_started_at = None

    def _log_command_issue(self, command_name: str):
        if not self._current_timing:
            return
        decision_ms = (time.perf_counter() - self._current_timing["loop_perf_start"]) * 1000
        logger.info(
            "[TIMING] cmd=%s state=%s target=%s action=%s person_det_ms=%.1f face_det_ms=%.1f frame_age_ms=%.1f decision_ms=%.1f",
            command_name,
            self.state.value,
            self._current_timing.get("target_kind", "none"),
            self.last_action,
            self._current_timing.get("person_det_ms", 0.0),
            self._current_timing.get("face_det_ms", 0.0),
            self._current_timing.get("frame_age_ms", 0.0),
            decision_ms,
        )

    def _record_timing_summary(self, person_det_ms: float, face_det_ms: float, loop_ms: float, frame_age_ms: float):
        self._timing_acc["count"] += 1
        self._timing_acc["person_det_ms"] += person_det_ms
        self._timing_acc["face_det_ms"] += face_det_ms
        self._timing_acc["loop_ms"] += loop_ms
        self._timing_acc["frame_age_ms"] += frame_age_ms

        now = time.time()
        if now - self._timing_last_log_ts < self.timing_log_interval_s:
            return

        count = max(self._timing_acc["count"], 1)
        logger.info(
            "[TIMING] summary state=%s loops=%d avg_person_det_ms=%.1f avg_face_det_ms=%.1f avg_loop_ms=%.1f avg_frame_age_ms=%.1f fps=%.1f",
            self.state.value,
            self._timing_acc["count"],
            self._timing_acc["person_det_ms"] / count,
            self._timing_acc["face_det_ms"] / count,
            self._timing_acc["loop_ms"] / count,
            self._timing_acc["frame_age_ms"] / count,
            self.fps,
        )
        self._timing_last_log_ts = now
        self._timing_acc = {
            "count": 0,
            "person_det_ms": 0.0,
            "face_det_ms": 0.0,
            "loop_ms": 0.0,
            "frame_age_ms": 0.0,
        }

    def _mark_engagement_start(self, frame_age_ms: float):
        if self._engagement_started_at is not None:
            return
        self._engagement_started_at = time.perf_counter()
        self._hold_logged = False
        logger.info("[TIMING] engage_start state=%s frame_age_ms=%.1f", self.state.value, frame_age_ms)

    def _reset_engagement(self, reason: str):
        self._hold_until = 0.0
        if self._engagement_started_at is None:
            return
        total_ms = (time.perf_counter() - self._engagement_started_at) * 1000
        logger.info("[TIMING] engage_reset reason=%s total_ms=%.1f", reason, total_ms)
        self._engagement_started_at = None
        self._hold_logged = False

    def _restore_home(self, reason: str):
        logger.info("[TIMING] restore_home reason=%s zoom_steps=%d", reason, self.zoom_steps)
        self.person_tracker.reset()
        self.zoom_steps = 0
        self.last_zoom_ts = 0.0
        self.ptz.goto_preset(self.cfg.stage1.home_preset)
        self.last_action = self.ptz.last_action
        self.settling_until = time.time() + self.cfg.stage1.restore_home_settle_s
        self._mark_settle_pending("restore_home")
        self._reset_engagement(reason)
        self._set_state(Stage1State.RESTORE_HOME, "restore_home")

    def _update_fps(self):
        self._fps_count += 1
        now = time.time()
        if now - self._fps_ts >= 1.0:
            self.fps = self._fps_count / (now - self._fps_ts)
            self._fps_count = 0
            self._fps_ts = now

    def _wait_first_frame(self):
        logger.info("Waiting for first RTSP frame...")
        for _ in range(100):
            frame = self.video.read()
            if frame is not None:
                logger.info("Video stream ready: %s", frame.shape)
                return frame
            time.sleep(0.1)
        raise RuntimeError("No RTSP frame received within 10 seconds")

    def _handle_signal(self, signum, _frame):
        logger.info("Signal %s received, stopping", signum)
        self.stop_flag = True

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
            self.person_tracker.reset()
            self.zoom_steps = 0
            self.ptz.goto_preset(self.cfg.stage1.home_preset)
            self.settling_until = time.time() + self.cfg.stage1.settle_after_move_s
            self._mark_settle_pending("home")
            self.last_action = self.ptz.last_action
            self._reset_engagement("home")
            self._set_state(Stage1State.SEARCH_PERSON, "home")
            return

        manual_map = {
            "left": lambda: self.ptz.pan_left(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed),
            "right": lambda: self.ptz.pan_right(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed),
            "up": lambda: self.ptz.tilt_up(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed),
            "down": lambda: self.ptz.tilt_down(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed),
            "zoom_in": lambda: self.ptz.zoom_in(self.cfg.stage1.zoom_pulse_s, self.cfg.stage1.zoom_speed),
            "zoom_out": lambda: self.ptz.zoom_out(self.cfg.stage1.zoom_pulse_s, self.cfg.stage1.zoom_speed),
            "stop": self.ptz.stop,
        }
        handler = manual_map.get(cmd)
        if handler is None:
            return
        handler()
        self.last_action = self.ptz.last_action
        if cmd in {"zoom_in", "zoom_out"}:
            self.settling_until = time.time() + self.cfg.stage1.settle_after_zoom_s
            self._mark_settle_pending(cmd)
        else:
            self.settling_until = time.time() + self.cfg.stage1.settle_after_move_s
            self._mark_settle_pending(cmd)

    def _nudge_towards(self, dx: float, dy: float, target_kind: str):
        command_name = None
        if target_kind == "face":
            tol_x = self.cfg.stage1.face_center_tolerance_ratio_x
            tol_y = self.cfg.stage1.face_center_tolerance_ratio_y
        else:
            tol_x = self.cfg.stage1.person_center_tolerance_ratio_x
            tol_y = self.cfg.stage1.person_center_tolerance_ratio_y

        if abs(dx) >= abs(dy) and abs(dx) > tol_x:
            if dx < 0:
                self.ptz.pan_left(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed)
                command_name = "pan_left"
            else:
                self.ptz.pan_right(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed)
                command_name = "pan_right"
        elif abs(dy) > tol_y:
            if dy < 0:
                self.ptz.tilt_up(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed)
                command_name = "tilt_up"
            else:
                self.ptz.tilt_down(self.cfg.stage1.move_pulse_s, self.cfg.stage1.move_speed)
                command_name = "tilt_down"
        self.last_action = self.ptz.last_action
        self.settling_until = time.time() + self.cfg.stage1.settle_after_move_s
        if command_name is not None:
            self._mark_settle_pending(command_name)
            self._log_command_issue(command_name)

    def _zoom_step(self):
        if self.zoom_steps >= self.cfg.stage1.max_zoom_steps:
            return
        self.ptz.zoom_in(self.cfg.stage1.zoom_pulse_s, self.cfg.stage1.zoom_speed)
        self.zoom_steps += 1
        self.last_zoom_ts = time.time()
        self.last_action = self.ptz.last_action
        self._set_state(Stage1State.ZOOM_STEP, "auto_zoom")
        self.settling_until = time.time() + self.cfg.stage1.settle_after_zoom_s
        self._mark_settle_pending("zoom_in")
        self._log_command_issue("zoom_in")

    def _control_target(self, target_kind: str, target_bbox, face_available: bool, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        cx, cy = bbox_center(target_bbox)
        dx = (cx - frame_w / 2.0) / max(frame_w, 1)
        dy = (cy - frame_h / 2.0) / max(frame_h, 1)
        width_ratio = (target_bbox[2] - target_bbox[0]) / max(frame_w, 1)

        if target_kind == "face":
            tol_x = self.cfg.stage1.face_center_tolerance_ratio_x
            tol_y = self.cfg.stage1.face_center_tolerance_ratio_y
            centered = abs(dx) <= tol_x and abs(dy) <= tol_y
            zoom_ready = (
                abs(dx) <= self.cfg.stage1.face_zoom_ready_ratio_x
                and abs(dy) <= self.cfg.stage1.face_zoom_ready_ratio_y
            )
            if centered and self.zoom_steps >= self.cfg.stage1.max_zoom_steps:
                self._set_state(Stage1State.HOLD_FACE, "face_centered")
                if self._hold_until <= 0.0:
                    self._hold_until = time.time() + self.cfg.stage1.hold_after_max_zoom_s
                    if self._engagement_started_at is not None and not self._hold_logged:
                        total_ms = (time.perf_counter() - self._engagement_started_at) * 1000
                        logger.info(
                            "[TIMING] hold_face total_ms=%.1f zoom_steps=%d face_ratio=%.3f",
                            total_ms,
                            self.zoom_steps,
                            width_ratio,
                        )
                        self._hold_logged = True
            elif zoom_ready and width_ratio < self.cfg.stage1.face_size_hold_ratio:
                if (time.time() - self.last_zoom_ts) >= self.cfg.stage1.min_zoom_interval_s:
                    logger.info(
                        "[TIMING] zoom_ready target=face dx=%.3f dy=%.3f face_ratio=%.3f",
                        dx,
                        dy,
                        width_ratio,
                    )
                    self._zoom_step()
                else:
                    self._set_state(Stage1State.CENTER_FACE, "face_wait_zoom_interval")
            elif not centered:
                self._hold_until = 0.0
                self._set_state(Stage1State.CENTER_FACE, "face_offset")
                self._nudge_towards(dx, dy, "face")
            else:
                self._hold_until = 0.0
                self._set_state(Stage1State.CENTER_FACE, "face_wait_zoom")
        else:
            tol_x = self.cfg.stage1.person_center_tolerance_ratio_x
            tol_y = self.cfg.stage1.person_center_tolerance_ratio_y
            centered = abs(dx) <= tol_x and abs(dy) <= tol_y
            self._hold_until = 0.0
            if not centered:
                self._set_state(Stage1State.CENTER_PERSON, "person_offset")
                self._nudge_towards(dx, dy, "person")
            elif face_available:
                self._set_state(Stage1State.CENTER_FACE, "face_available")
            elif (time.time() - self.last_zoom_ts) >= self.cfg.stage1.min_zoom_interval_s:
                self._zoom_step()
            else:
                self._set_state(Stage1State.CENTER_PERSON, "person_wait_zoom")

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
            default_move_speed=self.cfg.stage1.move_speed,
            default_zoom_speed=self.cfg.stage1.zoom_speed,
        )

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("Stage1 starting")
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
                    time.sleep(self.cfg.stage1.loop_sleep_s)
                    continue

                self._update_fps()
                self._maybe_log_settle_done()
                if self.state == Stage1State.HOLD_FACE and self._hold_until > 0.0 and time.time() >= self._hold_until:
                    self._restore_home("hold_timeout")
                    time.sleep(self.cfg.stage1.loop_sleep_s)
                    continue
                if self.state == Stage1State.HOLD_FACE and self._hold_until > 0.0 and time.time() < self._hold_until:
                    self.last_action = "hold_face_wait"
                    time.sleep(self.cfg.stage1.loop_sleep_s)
                    continue
                if self.state == Stage1State.RESTORE_HOME:
                    if time.time() >= self.settling_until:
                        logger.info("[TIMING] restore_home_done home_preset=%d", self.cfg.stage1.home_preset)
                        self._set_state(Stage1State.SEARCH_PERSON, "restore_done")
                    time.sleep(self.cfg.stage1.loop_sleep_s)
                    continue
                frame_age_ms = 0.0
                if self.video.timestamp > 0:
                    frame_age_ms = max(0.0, (time.time() - self.video.timestamp) * 1000)
                cmd = self.web.poll_command() if self.web is not None else None
                if cmd:
                    self._handle_command(cmd)

                person_det_started = time.perf_counter()
                person_dets = self.person_detector.detect(frame)
                person_det_ms = (time.perf_counter() - person_det_started) * 1000
                tracked_person = self.person_tracker.update(person_dets, frame.shape)
                person_visible = len(person_dets) > 0

                face_det_ms = 0.0
                face_dets = []
                if person_visible or self.state in {
                    Stage1State.CENTER_FACE,
                    Stage1State.HOLD_FACE,
                }:
                    face_det_started = time.perf_counter()
                    face_dets = self.face_detector.detect(frame)
                    face_det_ms = (time.perf_counter() - face_det_started) * 1000
                face_target = SingleTargetSelector.pick_face(
                    face_dets,
                    frame.shape,
                    tracked_person.bbox if tracked_person and person_visible else None,
                )

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
                    pass
                elif tracked_person is None and face_target is None:
                    if self.state == Stage1State.SEARCH_PERSON:
                        self.last_action = "waiting_for_person"
                    elif time.time() < self.settling_until:
                        self.last_action = "settling_wait_target"
                    else:
                        self._set_state(Stage1State.LOST_TARGET, "no_target")
                        self._restore_home("lost_target")
                        time.sleep(self.cfg.stage1.loop_sleep_s)
                        continue
                else:
                    self._mark_engagement_start(frame_age_ms)
                    if face_target is not None:
                        face_ratio = (face_target.bbox[2] - face_target.bbox[0]) / max(frame.shape[1], 1)
                        if face_ratio >= self.cfg.stage1.face_size_switch_ratio:
                            target_kind = "face"
                            target_bbox = face_target.bbox
                        else:
                            target_kind = "person"
                            target_bbox = tracked_person.bbox if tracked_person is not None else face_target.bbox
                    else:
                        target_kind = "person"
                        target_bbox = tracked_person.bbox
                    self._current_timing["target_kind"] = target_kind
                    if time.time() >= self.settling_until:
                        dx, dy, target_ratio = self._control_target(
                            target_kind,
                            target_bbox,
                            face_target is not None,
                            frame.shape,
                        )
                    else:
                        cx, cy = bbox_center(target_bbox)
                        dx = (cx - frame.shape[1] / 2.0) / max(frame.shape[1], 1)
                        dy = (cy - frame.shape[0] / 2.0) / max(frame.shape[0], 1)
                        target_ratio = (target_bbox[2] - target_bbox[0]) / max(frame.shape[1], 1)

                debug = {
                    "dx": f"{dx:+.3f}",
                    "dy": f"{dy:+.3f}",
                    "target_ratio": f"{target_ratio:.3f}",
                    "zoom_steps": self.zoom_steps,
                    "last_action": self.last_action,
                    "paused": self.paused,
                    "settling": time.time() < self.settling_until,
                    "fps": f"{self.fps:.1f}",
                }
                if self.web is not None:
                    annotated = self.overlay.render(
                        frame,
                        state=self.state.value,
                        person_dets=person_dets,
                        face_dets=face_dets,
                        tracked_person=tracked_person if person_visible else None,
                        target_kind=target_kind,
                        target_bbox=target_bbox,
                        debug=debug,
                        draw_all=self.cfg.stage1.debug_draw_all,
                    )
                    self.web.show(annotated)
                loop_ms = (time.perf_counter() - loop_perf_start) * 1000
                self._record_timing_summary(
                    person_det_ms=person_det_ms,
                    face_det_ms=face_det_ms,
                    loop_ms=loop_ms,
                    frame_age_ms=frame_age_ms,
                )
                time.sleep(self.cfg.stage1.loop_sleep_s)
        finally:
            logger.info("Stage1 shutting down")
            try:
                self.video.stop()
            except Exception:
                pass
            try:
                self.ptz.goto_preset(self.cfg.stage1.home_preset)
                time.sleep(self.cfg.stage1.restore_home_settle_s)
            except Exception:
                pass
            try:
                self.ptz.logout()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Stage1 single-static runtime")
    parser.add_argument("--common-config", default="config/common.yaml")
    parser.add_argument("--stage-config", default="config/stage1_single_static.yaml")
    parser.add_argument("--no-web", action="store_true", help="Disable MJPEG web streaming to reduce debug overhead")
    parser.add_argument("--timing-log-interval", type=float, default=1.0, help="Seconds between timing summary logs")
    args = parser.parse_args()

    app = Stage1SingleStaticApp(
        args.common_config,
        args.stage_config,
        enable_web=not args.no_web,
        timing_log_interval_s=args.timing_log_interval,
    )
    app.run()


if __name__ == "__main__":
    main()
