"""Hikvision PTZ business wrapper."""

import logging
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
                 user: str, password: str, channel: int):
        self.sdk = sdk
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.channel = channel
        self.user_id: Optional[int] = None

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
        """Check if device supports 3D positioning (PTZZoomIn)."""
        in_buf = b"<DeviceAbility>\r\n</DeviceAbility>"
        out_buf = (c_char * 8192)()
        ok = self.sdk.sdk.NET_DVR_GetDeviceAbility(
            self.user_id, self.DEVICE_ABILITY_INFO,
            in_buf, len(in_buf),
            out_buf, sizeof(out_buf),
        )
        if not ok:
            err = self.sdk.sdk.NET_DVR_GetLastError()
            logger.error(f"GetDeviceAbility failed, error={err}")
            return False

        result = out_buf.value.decode("utf-8", errors="replace")
        has_zoom = "<PTZZoomIn>" in result
        logger.info(f"3D positioning (PTZZoomIn) supported: {has_zoom}")
        return has_zoom

    def goto_preset(self, preset_id: int) -> bool:
        """Move PTZ to a preset position."""
        ok = self.sdk.sdk.NET_DVR_PTZPreset_Other(
            self.user_id, self.channel, self.GOTO_PRESET, preset_id,
        )
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
        expanded = bbox_expand(bbox, expand, frame_w, frame_h)
        coords = bbox_to_point_frame(expanded, frame_w, frame_h)

        pf = NET_DVR_POINT_FRAME()
        pf.xTop = coords[0]
        pf.yTop = coords[1]
        pf.xBottom = coords[2]
        pf.yBottom = coords[3]
        pf.bCounter = 0

        ok = self.sdk.sdk.NET_DVR_PTZSelZoomIn_EX(
            self.user_id, self.channel, byref(pf),
        )
        if not ok:
            err = self.sdk.sdk.NET_DVR_GetLastError()
            logger.warning(f"zoom_to_bbox failed, error={err}")
        return ok

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
