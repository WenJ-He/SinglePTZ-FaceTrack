"""Helpers for reading absolute PTZ coordinates and matching nearest presets."""

from __future__ import annotations

import ctypes as C
import math
import time
from dataclasses import dataclass

from src.hik_sdk import HikClient

NET_DVR_GET_PTZPOS = 293
NET_DVR_GET_PTZSCOPE = 294


class NET_DVR_PTZPOS(C.Structure):
    _fields_ = [
        ("wAction", C.c_ushort),
        ("wPanPos", C.c_ushort),
        ("wTiltPos", C.c_ushort),
        ("wZoomPos", C.c_ushort),
    ]


class NET_DVR_PTZSCOPE(C.Structure):
    _fields_ = [
        ("wPanPosMin", C.c_ushort),
        ("wPanPosMax", C.c_ushort),
        ("wTiltPosMin", C.c_ushort),
        ("wTiltPosMax", C.c_ushort),
        ("wZoomPosMin", C.c_ushort),
        ("wZoomPosMax", C.c_ushort),
    ]


@dataclass
class PtzCoord:
    pan: float
    tilt: float
    zoom: float
    raw_pan: int
    raw_tilt: int
    raw_zoom: int


@dataclass
class PtzScope:
    pan_min: float
    pan_max: float
    tilt_min: float
    tilt_max: float
    zoom_min: float
    zoom_max: float


def decode_bcd_tenth(value: int) -> float:
    text = f"{int(value) & 0xFFFF:04X}"
    if all(ch.isdigit() for ch in text):
        return int(text) / 10.0
    return int(value) / 10.0


def _configure_get_dvr_config(sdk) -> None:
    sdk.NET_DVR_GetDVRConfig.restype = C.c_bool
    sdk.NET_DVR_GetDVRConfig.argtypes = [
        C.c_long,
        C.c_uint32,
        C.c_long,
        C.c_void_p,
        C.c_uint32,
        C.POINTER(C.c_uint32),
    ]


def _fetch_dvr_config(client: HikClient, command: int, buffer, channel: int) -> int:
    _configure_get_dvr_config(client.sdk)
    attempts = [channel]
    if channel != 0:
        attempts.append(0)

    last_error = None
    for ch in attempts:
        returned = C.c_uint32(0)
        ok = client.sdk.NET_DVR_GetDVRConfig(
            client.user_id,
            command,
            ch,
            C.byref(buffer),
            C.sizeof(buffer),
            C.byref(returned),
        )
        if ok:
            return ch
        last_error = client.get_last_error()
    raise RuntimeError(f"NET_DVR_GetDVRConfig command={command} failed, err={last_error}")


def get_ptz_coord(client: HikClient, channel: int) -> tuple[PtzCoord, int]:
    buf = NET_DVR_PTZPOS()
    used_channel = _fetch_dvr_config(client, NET_DVR_GET_PTZPOS, buf, channel)
    coord = PtzCoord(
        pan=decode_bcd_tenth(buf.wPanPos),
        tilt=decode_bcd_tenth(buf.wTiltPos),
        zoom=decode_bcd_tenth(buf.wZoomPos),
        raw_pan=int(buf.wPanPos),
        raw_tilt=int(buf.wTiltPos),
        raw_zoom=int(buf.wZoomPos),
    )
    return coord, used_channel


def get_ptz_scope(client: HikClient, channel: int) -> tuple[PtzScope | None, int | None]:
    buf = NET_DVR_PTZSCOPE()
    try:
        used_channel = _fetch_dvr_config(client, NET_DVR_GET_PTZSCOPE, buf, channel)
    except Exception:
        return None, None
    scope = PtzScope(
        pan_min=decode_bcd_tenth(buf.wPanPosMin),
        pan_max=decode_bcd_tenth(buf.wPanPosMax),
        tilt_min=decode_bcd_tenth(buf.wTiltPosMin),
        tilt_max=decode_bcd_tenth(buf.wTiltPosMax),
        zoom_min=decode_bcd_tenth(buf.wZoomPosMin),
        zoom_max=decode_bcd_tenth(buf.wZoomPosMax),
    )
    return scope, used_channel


def capture_preset_coords(
    client: HikClient,
    preset_ids: list[int],
    settle_s: float,
    channel: int,
) -> dict[int, PtzCoord]:
    preset_coords: dict[int, PtzCoord] = {}
    for preset_id in preset_ids:
        if not client.goto_preset(preset_id):
            raise RuntimeError(f"goto_preset({preset_id}) failed, err={client.get_last_error()}")
        time.sleep(max(settle_s, 0.0))
        coord, _ = get_ptz_coord(client, channel)
        preset_coords[preset_id] = coord
    return preset_coords


def circular_diff(a: float, b: float, period: float) -> float:
    if period <= 0.0:
        return abs(a - b)
    delta = abs(a - b) % period
    return min(delta, period - delta)


def preset_distance(current: PtzCoord, preset: PtzCoord, scope: PtzScope | None) -> float:
    pan_range = 360.0
    tilt_range = 90.0
    zoom_range = 1.0

    if scope is not None:
        pan_range = max(scope.pan_max - scope.pan_min, pan_range)
        tilt_range = max(scope.tilt_max - scope.tilt_min, 1.0)
        zoom_range = max(scope.zoom_max - scope.zoom_min, 1.0)

    pan_term = circular_diff(current.pan, preset.pan, pan_range) / max(pan_range, 1e-6)
    tilt_term = abs(current.tilt - preset.tilt) / max(tilt_range, 1e-6)
    zoom_term = abs(current.zoom - preset.zoom) / max(zoom_range, 1e-6)
    return math.sqrt(pan_term * pan_term + tilt_term * tilt_term + zoom_term * zoom_term)


def nearest_preset(current: PtzCoord, preset_coords: dict[int, PtzCoord], scope: PtzScope | None) -> tuple[int, float]:
    ranked = sorted(
        ((preset_id, preset_distance(current, coord, scope)) for preset_id, coord in preset_coords.items()),
        key=lambda item: item[1],
    )
    return ranked[0]
