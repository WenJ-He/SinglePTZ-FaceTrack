"""Minimal HCNetSDK wrapper and PTZ command constants."""

import ctypes as C
import logging
import os
import sys
from ctypes import POINTER, Structure, byref, c_bool, c_byte, c_char, c_int, c_long, c_uint, c_uint32, c_ushort, c_void_p
from typing import Optional

logger = logging.getLogger("app")

ZOOM_IN = 11
ZOOM_OUT = 12
TILT_UP = 21
TILT_DOWN = 22
PAN_LEFT = 23
PAN_RIGHT = 24


class NET_DVR_USER_LOGIN_INFO(Structure):
    _fields_ = [
        ("sDeviceAddress", c_char * 129),
        ("byUseTransport", c_byte),
        ("wPort", c_ushort),
        ("sUserName", c_char * 64),
        ("sPassword", c_char * 64),
        ("cbLoginResult", c_void_p),
        ("pUser", c_void_p),
        ("bUseAsynLogin", c_int),
        ("byRes2", c_byte * 128),
    ]


class NET_DVR_DEVICEINFO_V40(Structure):
    _fields_ = [("struDeviceV30", c_byte * 448)]


class HikClient:
    """Minimal wrapper around HCNetSDK for login, PTZ control, and presets."""

    def __init__(self, lib_dir: str, ip: str, port: int, user: str, password: str, channel: int):
        self.lib_dir = os.path.abspath(lib_dir)
        self._sdk = None
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.channel = channel
        self.user_id: Optional[int] = None

    @property
    def sdk(self):
        if self._sdk is None:
            self._load_sdk()
        return self._sdk

    def _load_sdk(self):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if self.lib_dir not in existing:
            os.environ["LD_LIBRARY_PATH"] = self.lib_dir + ":" + existing
        try:
            sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)
        except AttributeError:
            pass

        lib_path = os.path.join(self.lib_dir, "libhcnetsdk.so")
        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"SDK library not found: {lib_path}")

        sdk = C.CDLL(lib_path, mode=C.RTLD_GLOBAL)
        sdk.NET_DVR_Init.restype = c_bool
        sdk.NET_DVR_Init.argtypes = []
        sdk.NET_DVR_Cleanup.restype = c_bool
        sdk.NET_DVR_Cleanup.argtypes = []
        sdk.NET_DVR_Login_V40.restype = c_long
        sdk.NET_DVR_Login_V40.argtypes = [
            POINTER(NET_DVR_USER_LOGIN_INFO),
            POINTER(NET_DVR_DEVICEINFO_V40),
        ]
        sdk.NET_DVR_Logout.restype = c_bool
        sdk.NET_DVR_Logout.argtypes = [c_long]
        sdk.NET_DVR_GetLastError.restype = c_uint32
        sdk.NET_DVR_GetLastError.argtypes = []
        sdk.NET_DVR_PTZControl_Other.restype = c_bool
        sdk.NET_DVR_PTZControl_Other.argtypes = [c_long, c_int, c_uint, c_uint]
        sdk.NET_DVR_PTZControlWithSpeed_Other.restype = c_bool
        sdk.NET_DVR_PTZControlWithSpeed_Other.argtypes = [c_long, c_int, c_uint, c_uint, c_uint]
        sdk.NET_DVR_PTZPreset_Other.restype = c_bool
        sdk.NET_DVR_PTZPreset_Other.argtypes = [c_long, c_int, c_uint, c_uint]
        self._sdk = sdk

    def login(self) -> None:
        self.sdk.NET_DVR_Init()
        login_info = NET_DVR_USER_LOGIN_INFO()
        login_info.sDeviceAddress = self.ip.encode("utf-8")
        login_info.wPort = self.port
        login_info.sUserName = self.user.encode("utf-8")
        login_info.sPassword = self.password.encode("utf-8")
        login_info.bUseAsynLogin = 0

        device_info = NET_DVR_DEVICEINFO_V40()
        self.user_id = self.sdk.NET_DVR_Login_V40(byref(login_info), byref(device_info))
        if self.user_id < 0:
            err = self.sdk.NET_DVR_GetLastError()
            self.sdk.NET_DVR_Cleanup()
            raise RuntimeError(f"SDK login failed, error={err}")
        logger.info("SDK login success, user_id=%s", self.user_id)

    def logout(self) -> None:
        if self.user_id is not None and self.user_id >= 0:
            self.sdk.NET_DVR_Logout(self.user_id)
            logger.info("SDK logout")
        self.sdk.NET_DVR_Cleanup()

    def get_last_error(self) -> int:
        return int(self.sdk.NET_DVR_GetLastError())

    def ptz_control(self, command: int, stop: bool = False, speed: Optional[int] = None) -> bool:
        if self.user_id is None or self.user_id < 0:
            raise RuntimeError("PTZ control requested before login")

        stop_flag = 1 if stop else 0
        if speed is None:
            ok = self.sdk.NET_DVR_PTZControl_Other(self.user_id, self.channel, command, stop_flag)
        else:
            ok = self.sdk.NET_DVR_PTZControlWithSpeed_Other(
                self.user_id, self.channel, command, stop_flag, speed
            )

        if not ok:
            logger.warning(
                "PTZ control failed: cmd=%s stop=%s speed=%s err=%s",
                command, stop_flag, speed, self.get_last_error(),
            )
        return bool(ok)

    def goto_preset(self, preset_id: int) -> bool:
        if self.user_id is None or self.user_id < 0:
            raise RuntimeError("goto_preset requested before login")
        ok = self.sdk.NET_DVR_PTZPreset_Other(self.user_id, self.channel, 39, preset_id)
        if not ok:
            logger.warning("goto_preset(%s) failed, err=%s", preset_id, self.get_last_error())
        return bool(ok)
