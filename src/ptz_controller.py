"""PTZ controller helpers for pulse and continuous closed-loop motion."""

import threading
import time

from src.hik_sdk import HikClient, PAN_LEFT, PAN_RIGHT, TILT_DOWN, TILT_UP, ZOOM_IN, ZOOM_OUT


class HikManualPtzController:
    """Expose manual PTZ pulses and non-blocking drive helpers."""

    def __init__(self, client: HikClient, default_move_speed: int = 4, default_zoom_speed: int = 4):
        self.client = client
        self.default_move_speed = default_move_speed
        self.default_zoom_speed = default_zoom_speed
        self.last_action = "idle"
        self._last_command = None
        self._last_speed = None
        self._deadline = 0.0
        self._lock = threading.Lock()

    def login(self) -> None:
        self.client.login()

    def logout(self) -> None:
        self.client.logout()

    def goto_preset(self, preset_id: int) -> bool:
        with self._lock:
            self._stop_locked(label="stop(preset)")
            ok = self.client.goto_preset(preset_id)
            self.last_action = f"goto_preset({preset_id})"
            self._last_command = None
            self._last_speed = None
            self._deadline = 0.0
            return ok

    def stop(self) -> bool:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self, label: str = "stop") -> bool:
        if self._last_command is None:
            self.last_action = f"{label}(noop)"
            return True
        ok = self.client.ptz_control(self._last_command, stop=True, speed=self._last_speed)
        self.last_action = label
        self._last_command = None
        self._last_speed = None
        self._deadline = 0.0
        return bool(ok)

    def _start_locked(self, command: int, speed: int, label: str) -> bool:
        if self._last_command == command and self._last_speed == speed:
            return True
        if self._last_command is not None:
            self.client.ptz_control(self._last_command, stop=True, speed=self._last_speed)
        ok = self.client.ptz_control(command, stop=False, speed=speed)
        if not ok:
            self.last_action = f"{label}(failed)"
            return False
        self._last_command = command
        self._last_speed = speed
        return True

    def drive(self, command: int, duration_s: float | None, speed: int, label: str) -> bool:
        duration_s = max(float(duration_s or 0.0), 0.0)
        with self._lock:
            same_command = self._last_command == command and self._last_speed == speed
            if not self._start_locked(command, speed, label):
                return False
            now = time.time()
            if duration_s > 0.0:
                new_deadline = now + duration_s
                self._deadline = max(self._deadline, new_deadline) if same_command else new_deadline
                remaining = max(self._deadline - now, 0.0)
                self.last_action = f"{label}@{speed}:{remaining:.2f}s"
            else:
                self._deadline = 0.0
                self.last_action = f"{label}@{speed}:hold"
            return True

    def tick(self) -> bool:
        with self._lock:
            if self._last_command is None or self._deadline <= 0.0:
                return True
            if time.time() < self._deadline:
                return True
            return self._stop_locked(label="stop(auto)")

    def _pulse(self, command: int, duration_s: float, speed: int, label: str) -> bool:
        if not self.drive(command, duration_s, speed, label):
            return False
        time.sleep(max(duration_s, 0.0))
        return self.stop()

    def pan_left_async(self, duration_s: float, speed: int | None = None) -> bool:
        return self.drive(PAN_LEFT, duration_s, speed or self.default_move_speed, "pan_left")

    def pan_right_async(self, duration_s: float, speed: int | None = None) -> bool:
        return self.drive(PAN_RIGHT, duration_s, speed or self.default_move_speed, "pan_right")

    def tilt_up_async(self, duration_s: float, speed: int | None = None) -> bool:
        return self.drive(TILT_UP, duration_s, speed or self.default_move_speed, "tilt_up")

    def tilt_down_async(self, duration_s: float, speed: int | None = None) -> bool:
        return self.drive(TILT_DOWN, duration_s, speed or self.default_move_speed, "tilt_down")

    def zoom_in_async(self, duration_s: float, speed: int | None = None) -> bool:
        return self.drive(ZOOM_IN, duration_s, speed or self.default_zoom_speed, "zoom_in")

    def zoom_out_async(self, duration_s: float, speed: int | None = None) -> bool:
        return self.drive(ZOOM_OUT, duration_s, speed or self.default_zoom_speed, "zoom_out")

    def pan_left(self, duration_s: float, speed: int | None = None) -> bool:
        return self._pulse(PAN_LEFT, duration_s, speed or self.default_move_speed, "pan_left")

    def pan_right(self, duration_s: float, speed: int | None = None) -> bool:
        return self._pulse(PAN_RIGHT, duration_s, speed or self.default_move_speed, "pan_right")

    def tilt_up(self, duration_s: float, speed: int | None = None) -> bool:
        return self._pulse(TILT_UP, duration_s, speed or self.default_move_speed, "tilt_up")

    def tilt_down(self, duration_s: float, speed: int | None = None) -> bool:
        return self._pulse(TILT_DOWN, duration_s, speed or self.default_move_speed, "tilt_down")

    def zoom_in(self, duration_s: float, speed: int | None = None) -> bool:
        return self._pulse(ZOOM_IN, duration_s, speed or self.default_zoom_speed, "zoom_in")

    def zoom_out(self, duration_s: float, speed: int | None = None) -> bool:
        return self._pulse(ZOOM_OUT, duration_s, speed or self.default_zoom_speed, "zoom_out")
