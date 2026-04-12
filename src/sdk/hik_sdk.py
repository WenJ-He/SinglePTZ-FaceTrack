"""Hikvision SDK ctypes bindings for HCNetSDK."""

import ctypes as C
import os
import sys
from ctypes import (
    c_int, c_long, c_uint, c_char, c_byte, c_ushort,
    c_void_p, c_bool, c_uint32, c_char_p, POINTER, Structure, byref,
)


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
    _fields_ = [
        ("struDeviceV30", c_byte * 448),
    ]


class NET_DVR_POINT_FRAME(Structure):
    _fields_ = [
        ("xTop", c_int),
        ("yTop", c_int),
        ("xBottom", c_int),
        ("yBottom", c_int),
        ("bCounter", c_int),
    ]


class HikSDK:
    """Lazy-loaded ctypes binding for libhcnetsdk.so."""

    def __init__(self, lib_dir: str):
        self.lib_dir = os.path.abspath(lib_dir)
        self._sdk = None

    def load(self):
        """Load the SDK shared library."""
        if self._sdk is not None:
            return self._sdk

        # Set LD_LIBRARY_PATH before loading
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if self.lib_dir not in existing:
            os.environ["LD_LIBRARY_PATH"] = self.lib_dir + ":" + existing
        # Also set for dlopen
        try:
            sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)
        except AttributeError:
            pass

        lib_path = os.path.join(self.lib_dir, "libhcnetsdk.so")
        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"SDK library not found: {lib_path}")

        self._sdk = C.CDLL(lib_path, mode=C.RTLD_GLOBAL)
        self._declare_functions()
        return self._sdk

    @property
    def sdk(self):
        if self._sdk is None:
            raise RuntimeError("SDK not loaded. Call load() first.")
        return self._sdk

    def _declare_functions(self):
        s = self._sdk

        # Init / Cleanup
        s.NET_DVR_Init.restype = c_bool
        s.NET_DVR_Init.argtypes = []

        s.NET_DVR_Cleanup.restype = c_bool
        s.NET_DVR_Cleanup.argtypes = []

        # Login / Logout
        s.NET_DVR_Login_V40.restype = c_long
        s.NET_DVR_Login_V40.argtypes = [
            POINTER(NET_DVR_USER_LOGIN_INFO),
            POINTER(NET_DVR_DEVICEINFO_V40),
        ]

        s.NET_DVR_Logout.restype = c_bool
        s.NET_DVR_Logout.argtypes = [c_long]

        # Error
        s.NET_DVR_GetLastError.restype = c_uint32
        s.NET_DVR_GetLastError.argtypes = []

        # PTZ Preset: SET=8, CLE=9, GOTO=39
        s.NET_DVR_PTZPreset_Other.restype = c_bool
        s.NET_DVR_PTZPreset_Other.argtypes = [c_long, c_int, c_uint, c_uint]

        # 3D Positioning
        s.NET_DVR_PTZSelZoomIn_EX.restype = c_bool
        s.NET_DVR_PTZSelZoomIn_EX.argtypes = [
            c_long, c_int, POINTER(NET_DVR_POINT_FRAME),
        ]

        # PTZ Control: ZOOM_IN=11, ZOOM_OUT=12
        s.NET_DVR_PTZControl_Other.restype = c_bool
        s.NET_DVR_PTZControl_Other.argtypes = [c_long, c_int, c_uint, c_uint]

        # PTZ Control with Speed
        s.NET_DVR_PTZControlWithSpeed_Other.restype = c_bool
        s.NET_DVR_PTZControlWithSpeed_Other.argtypes = [
            c_long, c_int, c_uint, c_uint, c_uint,
        ]

        # Cruise
        s.NET_DVR_PTZCruise_Other.restype = c_bool
        s.NET_DVR_PTZCruise_Other.argtypes = [
            c_long, c_int, c_uint, c_byte, c_byte, c_ushort,
        ]

        # Record
        s.NET_DVR_StartDVRRecord.restype = c_bool
        s.NET_DVR_StartDVRRecord.argtypes = [c_long, c_int, c_int]

        s.NET_DVR_StopDVRRecord.restype = c_bool
        s.NET_DVR_StopDVRRecord.argtypes = [c_long, c_int]

        # Device Ability
        # dwAbilityType = 0x011 for DEVICE_ABILITY_INFO
        s.NET_DVR_GetDeviceAbility.restype = c_bool
        s.NET_DVR_GetDeviceAbility.argtypes = [
            c_long, c_int, c_char_p, c_uint, c_char_p, c_uint,
        ]
