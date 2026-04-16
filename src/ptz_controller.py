"""Manual pulse-based PTZ controller for early-stage closed-loop tuning."""

import threading
import time

from src.hik_sdk import HikClient, PAN_LEFT, PAN_RIGHT, TILT_DOWN, TILT_UP, ZOOM_IN, ZOOM_OUT


class HikManualPtzController:
    """Expose manual PTZ pulses instead of high-level bbox zoom logic."""

    def __init__(self, client: HikClient, default_move_speed: int = 4, default_zoom_speed: int = 4):
        self.client = client
        self.default_move_speed = default_move_speed
        self.default_zoom_speed = default_zoom_speed
        self.last_action = "idle"
        self._last_command = None
        self._lock = threading.Lock()

    def login(self) -> None:
        self.client.login()

    def logout(self) -> None:
        self.client.logout()

    def goto_preset(self, preset_id: int) -> bool:
        with self._lock:
            ok = self.client.goto_preset(preset_id)
            self.last_action = f"goto_preset({preset_id})"
            self._last_command = None
            return ok

    def stop(self) -> bool:
        with self._lock:
            if self._last_command is None:
                self.last_action = "stop(noop)"
                return True
            ok = self.client.ptz_control(self._last_command, stop=True)
            self.last_action = "stop"
            self._last_command = None
            return ok

    def _pulse(self, command: int, duration_s: float, speed: int, label: str) -> bool:
        with self._lock:
            ok = self.client.ptz_control(command, stop=False, speed=speed)
            if not ok:
                self.last_action = f"{label}(failed)"
                return False
            self._last_command = command
            self.last_action = f"{label}@{speed}:{duration_s:.2f}s"
            time.sleep(max(duration_s, 0.0))
            self.client.ptz_control(command, stop=True, speed=speed)
            self._last_command = None
            return True

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
