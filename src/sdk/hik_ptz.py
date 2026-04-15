"""Hikvision PTZ business wrapper."""

import logging
import time
from ctypes import sizeof
from typing import Optional, Tuple

from src.sdk.hik_sdk import (
    HikSDK, NET_DVR_USER_LOGIN_INFO, NET_DVR_DEVICEINFO_V40,
    NET_DVR_POINT_FRAME,
    c_long, c_int, c_uint, byref, c_char, c_byte, c_char_p,
)
from src.utils.geometry import bbox_expand, bbox_to_point_frame

logger = logging.getLogger("app")


class HikPTZ:
    """High-level PTZ operations using Hikvision SDK."""

    GOTO_PRESET = 39
    SET_PRESET = 8
    CLE_PRESET = 9
    DEVICE_ABILITY_INFO = 0x011

    def __init__(self, sdk: HikSDK, ip: str, port: int,
                 user: str, password: str, channel: int,
                 min_interval: float = 0.8,
                 isapi=None,
                 zoom_backend: str = "sdk"):
        self.sdk = sdk
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.channel = channel
        self.user_id: Optional[int] = None
        self._last_cmd_ts: float = 0.0
        self.min_interval = min_interval
        self.isapi = isapi
        self.zoom_backend = zoom_backend

    def login(self) -> None:
        """Login to device. Raises on failure."""
        self.sdk.load()
        self.sdk.sdk.NET_DVR_Init()

        login_info = NET_DVR_USER_LOGIN_INFO()
        login_info.sDeviceAddress = self.ip.encode("utf-8")
        login_info.wPort = self.port
        login_info.sUserName = self.user.encode("utf-8")
        login_info.sPassword = self.password.encode("utf-8")
        login_info.bUseAsynLogin = 0

        device_info = NET_DVR_DEVICEINFO_V40()

        self.user_id = self.sdk.sdk.NET_DVR_Login_V40(
            byref(login_info), byref(device_info),
        )

        if self.user_id < 0:
            err = self.sdk.sdk.NET_DVR_GetLastError()
            self.sdk.sdk.NET_DVR_Cleanup()
            raise RuntimeError(
                f"SDK login failed, error code: {err}"
            )

        logger.info(f"SDK login success, user_id={self.user_id}")

    def logout(self) -> None:
        """Logout from device."""
        if self.user_id is not None and self.user_id >= 0:
            self.sdk.sdk.NET_DVR_Logout(self.user_id)
            logger.info("SDK logout")
        self.sdk.sdk.NET_DVR_Cleanup()

    def check_3d_positioning(self) -> bool:
        """Check if device supports 3D positioning (PTZZoomIn).

        Tries multiple input XML formats for DEVICE_ABILITY_INFO(0x011).
        Returns True if PTZZoomIn is found in any response.
        If ability query fails entirely, returns True with a warning
        (device likely supports it, and actual calls will fail gracefully).
        """
        # Input XML formats to try (different SDK versions expect different formats)
        input_formats = [
            b"<PTZAbility></PTZAbility>",
            b"<DeviceAbility><PTZAbility></PTZAbility></DeviceAbility>",
            b"<DeviceAbility>\r\n</DeviceAbility>",
        ]

        for in_buf in input_formats:
            out_buf = (c_char * 65536)()
            ok = self.sdk.sdk.NET_DVR_GetDeviceAbility(
                self.user_id, self.DEVICE_ABILITY_INFO,
                in_buf, len(in_buf),
                out_buf, sizeof(out_buf),
            )
            if not ok:
                err = self.sdk.sdk.NET_DVR_GetLastError()
                logger.debug(
                    f"GetDeviceAbility with '{in_buf.decode()}' failed, "
                    f"error={err}"
                )
                continue

            result = out_buf.value.decode("utf-8", errors="replace")
            has_zoom = "<PTZZoomIn>" in result or "<PTZZoomIn/>" in result
            logger.info(f"3D positioning (PTZZoomIn) supported: {has_zoom}")
            if has_zoom:
                return True
            # Got a valid response but no PTZZoomIn
            logger.warning(f"Ability response (no PTZZoomIn): {result[:500]}")
            return False

        # All ability queries failed - warn but continue
        logger.warning(
            "GetDeviceAbility failed for all input formats. "
            "Assuming 3D positioning is supported; actual PTZSelZoomIn_EX "
            "calls will fail gracefully if not."
        )
        return True

    def goto_preset(self, preset_id: int) -> bool:
        """Move PTZ to a preset position."""
        self._wait_if_needed()
        ok = self.sdk.sdk.NET_DVR_PTZPreset_Other(
            self.user_id, self.channel, self.GOTO_PRESET, preset_id,
        )
        self._last_cmd_ts = time.time()
        if not ok:
            err = self.sdk.sdk.NET_DVR_GetLastError()
            logger.warning(f"goto_preset({preset_id}) failed, error={err}")
        return ok

    def zoom_to_bbox(self, bbox: Tuple[int, int, int, int],
                     frame_w: int, frame_h: int,
                     expand: float = 2.0) -> bool:
        """3D positioning: zoom to expanded bbox area.

        bbox: (x1, y1, x2, y2) in pixel coordinates
        expand: outward expansion ratio
        """
        self._wait_if_needed()
        expanded = bbox_expand(bbox, expand, frame_w, frame_h)

        # ISAPI path: use ptzDrag
        if self.zoom_backend == "isapi" and self.isapi is not None:
            return self._zoom_via_isapi(expanded, frame_w, frame_h)

        # SDK path: NET_DVR_PTZSelZoomIn_EX
        coords = bbox_to_point_frame(expanded, frame_w, frame_h)
        logger.info(
            f"  zoom_to_bbox[SDK]: bbox={bbox}, expanded={expanded}, "
            f"frame={frame_w}x{frame_h}, "
            f"normalized=[{coords[0]},{coords[1]},{coords[2]},{coords[3]}]/255"
        )

        pf = NET_DVR_POINT_FRAME()
        pf.xTop = coords[0]
        pf.yTop = coords[1]
        pf.xBottom = coords[2]
        pf.yBottom = coords[3]
        pf.bCounter = 0

        ok = self.sdk.sdk.NET_DVR_PTZSelZoomIn_EX(
            self.user_id, self.channel, byref(pf),
        )
        self._last_cmd_ts = time.time()

        err = self.sdk.sdk.NET_DVR_GetLastError()
        if not ok:
            logger.warning(
                f"  zoom_to_bbox SDK FAILED: ret={ok}, error={err}"
            )
        else:
            logger.info(
                f"  zoom_to_bbox SDK returned OK (last_err={err})"
            )
        return ok

    def _zoom_via_isapi(self, expanded: Tuple[int, int, int, int],
                        frame_w: int, frame_h: int) -> bool:
        """Zoom using ISAPI absolute positioning.

        Reads current PTZ position, calculates target pan/tilt/zoom
        from the bbox, and sends absolute positioning command.
        """
        if self.isapi is None:
            logger.warning("  ISAPI zoom: no ISAPI instance")
            return False

        # Get current position
        pos = self.isapi.get_ptz_status()
        if pos is None:
            logger.warning("  ISAPI zoom: cannot read current PTZ position")
            return False

        cur_az = pos["azimuth"]
        cur_el = pos["elevation"]
        cur_zoom = pos["absoluteZoom"]

        # Estimate FOV at zoom=1x (typical for Hikvision dome cameras)
        # These can be calibrated per-camera; defaults are conservative
        hfov_at_1x = getattr(self, '_hfov_1x', 58.0)  # degrees
        vfov_at_1x = getattr(self, '_vfov_1x', 33.0)

        # Current effective FOV
        cur_hfov = hfov_at_1x / max(cur_zoom, 0.1)
        cur_vfov = vfov_at_1x / max(cur_zoom, 0.1)

        # Bbox center offset from frame center (normalized -1..+1)
        cx = ((expanded[0] + expanded[2]) / 2 - frame_w / 2) / frame_w
        cy = ((expanded[1] + expanded[3]) / 2 - frame_h / 2) / frame_h

        # Angular offset
        d_az = cx * cur_hfov
        d_el = -cy * cur_vfov  # negative because y-down in image, elevation positive is up

        # Target zoom: fit bbox to ~85% of frame.
        # Person detection bboxes are tall (full body); the expand_ratio=1.5
        # further enlarges them. With 0.6 fill, expanded bbox can exceed frame
        # height, yielding barely any zoom. 0.85 ensures meaningful zoom while
        # leaving safe margins for capture tracking.
        fill = getattr(self, '_zoom_fill', 0.85)
        bbox_w = expanded[2] - expanded[0]
        bbox_h = expanded[3] - expanded[1]
        zoom_by_w = (frame_w * fill) / max(bbox_w, 1) * cur_zoom
        zoom_by_h = (frame_h * fill) / max(bbox_h, 1) * cur_zoom
        target_zoom = min(zoom_by_w, zoom_by_h, 20.0)  # cap at 20x

        target_az = cur_az + d_az
        target_el = cur_el + d_el

        logger.info(
            f"  zoom_to_bbox[ISAPI]: cur=({cur_az:.1f},{cur_el:.1f},z{cur_zoom:.1f}), "
            f"offset=({d_az:.2f},{d_el:.2f}), target=({target_az:.1f},{target_el:.1f},z{target_zoom:.1f})"
        )

        ok = self.isapi.ptz_absolute_zoom(target_az, target_el, target_zoom)
        self._last_cmd_ts = time.time()
        return ok

    def query_native_resolution(self) -> Optional[Tuple[int, int]]:
        """Query and cache native camera resolution via ISAPI."""
        if self.isapi is None:
            return None
        res = self.isapi.get_native_resolution()
        if res:
            self._native_resolution = res
        return res

    def _wait_if_needed(self) -> None:
        """Enforce minimum interval between PTZ commands (debounce)."""
        elapsed = time.time() - self._last_cmd_ts
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"PTZ debounce: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

    def start_record(self) -> bool:
        """Start manual recording."""
        ok = self.sdk.sdk.NET_DVR_StartDVRRecord(
            self.user_id, self.channel, 0,
        )
        if not ok:
            err = self.sdk.sdk.NET_DVR_GetLastError()
            logger.warning(f"start_record failed, error={err}")
        return ok

    def stop_record(self) -> bool:
        """Stop recording."""
        ok = self.sdk.sdk.NET_DVR_StopDVRRecord(
            self.user_id, self.channel,
        )
        if not ok:
            err = self.sdk.sdk.NET_DVR_GetLastError()
            logger.warning(f"stop_record failed, error={err}")
        return ok
