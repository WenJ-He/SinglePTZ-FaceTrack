#!/usr/bin/env python3
"""Probe PTZ absolute coordinates and map current pose to the nearest preset."""

from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.app_config import load_app_config
from src.hik_sdk import HikClient
from src.logger import setup_logger

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


def _decode_bcd_tenth(value: int) -> float:
    """HCNetSDK exposes PTZ values as packed hex digits representing decimal*10."""
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


def _fetch_dvr_config(client: HikClient, command: int, buffer, channel: int):
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
        pan=_decode_bcd_tenth(buf.wPanPos),
        tilt=_decode_bcd_tenth(buf.wTiltPos),
        zoom=_decode_bcd_tenth(buf.wZoomPos),
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
        pan_min=_decode_bcd_tenth(buf.wPanPosMin),
        pan_max=_decode_bcd_tenth(buf.wPanPosMax),
        tilt_min=_decode_bcd_tenth(buf.wTiltPosMin),
        tilt_max=_decode_bcd_tenth(buf.wTiltPosMax),
        zoom_min=_decode_bcd_tenth(buf.wZoomPosMin),
        zoom_max=_decode_bcd_tenth(buf.wZoomPosMax),
    )
    return scope, used_channel


def _circular_diff(a: float, b: float, period: float) -> float:
    if period <= 0.0:
        return abs(a - b)
    delta = abs(a - b) % period
    return min(delta, period - delta)


def _distance(current: PtzCoord, preset: PtzCoord, scope: PtzScope | None) -> float:
    pan_range = 360.0
    tilt_range = 90.0
    zoom_range = 1.0

    if scope is not None:
        pan_range = max(scope.pan_max - scope.pan_min, pan_range)
        tilt_range = max(scope.tilt_max - scope.tilt_min, 1.0)
        zoom_range = max(scope.zoom_max - scope.zoom_min, 1.0)

    pan_term = _circular_diff(current.pan, preset.pan, pan_range) / max(pan_range, 1e-6)
    tilt_term = abs(current.tilt - preset.tilt) / max(tilt_range, 1e-6)
    zoom_term = abs(current.zoom - preset.zoom) / max(zoom_range, 1e-6)
    return math.sqrt(pan_term * pan_term + tilt_term * tilt_term + zoom_term * zoom_term)


def capture_presets(client: HikClient, preset_ids: list[int], settle_s: float, channel: int) -> dict[int, PtzCoord]:
    preset_coords: dict[int, PtzCoord] = {}
    for preset_id in preset_ids:
        print(f"[STEP] goto preset={preset_id}")
        if not client.goto_preset(preset_id):
            raise RuntimeError(f"goto_preset({preset_id}) failed, err={client.get_last_error()}")
        time.sleep(max(settle_s, 0.0))
        coord, used_channel = get_ptz_coord(client, channel)
        preset_coords[preset_id] = coord
        print(
            "[PRESET] id=%d channel=%d pan=%.1f tilt=%.1f zoom=%.1f raw=(0x%04X,0x%04X,0x%04X)"
            % (
                preset_id,
                used_channel,
                coord.pan,
                coord.tilt,
                coord.zoom,
                coord.raw_pan,
                coord.raw_tilt,
                coord.raw_zoom,
            )
        )
    return preset_coords


def print_nearest(current: PtzCoord, preset_coords: dict[int, PtzCoord], scope: PtzScope | None) -> None:
    ranked = sorted(
        (
            (
                preset_id,
                _distance(current, preset_coord, scope),
                _circular_diff(current.pan, preset_coord.pan, max((scope.pan_max - scope.pan_min), 360.0) if scope else 360.0),
                abs(current.tilt - preset_coord.tilt),
                abs(current.zoom - preset_coord.zoom),
            )
            for preset_id, preset_coord in preset_coords.items()
        ),
        key=lambda item: item[1],
    )
    nearest = ranked[0]
    print(
        "[CURRENT] pan=%.1f tilt=%.1f zoom=%.1f nearest_preset=%d score=%.6f pan_diff=%.1f tilt_diff=%.1f zoom_diff=%.1f"
        % (
            current.pan,
            current.tilt,
            current.zoom,
            nearest[0],
            nearest[1],
            nearest[2],
            nearest[3],
            nearest[4],
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Probe PTZ absolute coordinates and nearest preset.")
    parser.add_argument("--common-config", default="config/common.yaml")
    parser.add_argument("--stage-config", default="config/stage2_single_moving.yaml")
    parser.add_argument("--preset-ids", default="1,2,3,4", help="Comma-separated preset ids to capture.")
    parser.add_argument("--settle-s", type=float, default=3.0, help="Seconds to wait after goto_preset.")
    parser.add_argument("--poll-interval-s", type=float, default=0.5, help="Polling interval after preset capture.")
    parser.add_argument("--samples", type=int, default=10, help="Number of current-position samples after capture. 0 means forever.")
    parser.add_argument("--json-out", default="", help="Optional file to write captured preset coordinates and scope.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_app_config(args.common_config, args.stage_config)
    setup_logger("app", cfg.log.level, None)

    preset_ids = [int(item.strip()) for item in args.preset_ids.split(",") if item.strip()]
    if not preset_ids:
        raise ValueError("No preset ids provided")

    client = HikClient(
        lib_dir=cfg.hik.sdk_lib_dir,
        ip=cfg.hik.ip,
        port=cfg.hik.port,
        user=cfg.hik.user,
        password=cfg.hik.password,
        channel=cfg.hik.channel,
    )

    try:
        client.login()

        scope, scope_channel = get_ptz_scope(client, cfg.hik.channel)
        if scope is None:
            print("[SCOPE] unavailable, nearest-preset distance will use fallback ranges")
        else:
            print(
                "[SCOPE] channel=%d pan=[%.1f, %.1f] tilt=[%.1f, %.1f] zoom=[%.1f, %.1f]"
                % (
                    scope_channel,
                    scope.pan_min,
                    scope.pan_max,
                    scope.tilt_min,
                    scope.tilt_max,
                    scope.zoom_min,
                    scope.zoom_max,
                )
            )

        current, used_channel = get_ptz_coord(client, cfg.hik.channel)
        print(
            "[PTZ] support confirmed via NET_DVR_GET_PTZPOS on channel=%d pan=%.1f tilt=%.1f zoom=%.1f"
            % (used_channel, current.pan, current.tilt, current.zoom)
        )

        preset_coords = capture_presets(client, preset_ids, args.settle_s, cfg.hik.channel)

        if args.json_out:
            payload = {
                "scope": asdict(scope) if scope is not None else None,
                "presets": {str(preset_id): asdict(coord) for preset_id, coord in preset_coords.items()},
            }
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[JSON] wrote {args.json_out}")

        print("[INFO] polling current PTZ position and nearest preset...")
        sample_count = 0
        while True:
            current, _ = get_ptz_coord(client, cfg.hik.channel)
            print_nearest(current, preset_coords, scope)
            sample_count += 1
            if args.samples > 0 and sample_count >= args.samples:
                break
            time.sleep(max(args.poll_interval_s, 0.0))
    finally:
        try:
            client.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
