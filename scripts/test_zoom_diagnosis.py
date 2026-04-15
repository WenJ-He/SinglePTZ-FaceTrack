#!/usr/bin/env python3
"""Diagnose zoom_to_bbox behavior: SDK vs ISAPI vs preview handle.

Usage:
    conda run -n single_ptz_facetrack python scripts/test_zoom_diagnosis.py
    conda run -n single_ptz_facetrack python scripts/test_zoom_diagnosis.py --backend isapi
    conda run -n single_ptz_facetrack python scripts/test_zoom_diagnosis.py --backend sdk
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.config import load_config
from src.utils.logger import setup_logger
from src.sdk.hik_sdk import HikSDK
from src.sdk.hik_ptz import HikPTZ
from src.sdk.hik_isapi import HikISAPI
from src.utils.geometry import bbox_expand, bbox_to_point_frame
from src.sdk.hik_sdk import NET_DVR_POINT_FRAME, byref


def test_sdk_zoom(ptz: HikPTZ, x1, y1, x2, y2, frame_w, frame_h):
    """Test SDK NET_DVR_PTZSelZoomIn_EX with detailed logging."""
    print(f"\n{'='*60}")
    print(f"Testing SDK zoom_to_bbox")
    print(f"  bbox=({x1},{y1},{x2},{y2}), frame={frame_w}x{frame_h}")

    expanded = bbox_expand((x1, y1, x2, y2), 1.5, frame_w, frame_h)
    coords = bbox_to_point_frame(expanded, frame_w, frame_h)
    print(f"  expanded={expanded}")
    print(f"  normalized=[{coords[0]},{coords[1]},{coords[2]},{coords[3]}]/255")

    pf = NET_DVR_POINT_FRAME()
    pf.xTop = coords[0]
    pf.yTop = coords[1]
    pf.xBottom = coords[2]
    pf.yBottom = coords[3]
    pf.bCounter = 0

    print(f"  Calling NET_DVR_PTZSelZoomIn_EX...")
    ok = ptz.sdk.sdk.NET_DVR_PTZSelZoomIn_EX(
        ptz.user_id, ptz.channel, byref(pf),
    )
    err = ptz.sdk.sdk.NET_DVR_GetLastError()
    print(f"  Result: ok={ok}, GetLastError={err}")
    return ok


def test_isapi_zoom(isapi: HikISAPI, x1, y1, x2, y2, screen_w, screen_h):
    """Test ISAPI ptzDrag."""
    print(f"\n{'='*60}")
    print(f"Testing ISAPI ptzDrag zoom")
    print(f"  bbox=({x1},{y1},{x2},{y2}), screen={screen_w}x{screen_h}")
    ok = isapi.ptz_drag_zoom(x1, y1, x2, y2, screen_w, screen_h)
    print(f"  Result: ok={ok}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Zoom diagnosis tool")
    parser.add_argument("--backend", default="both",
                        choices=["sdk", "isapi", "both"],
                        help="Which backend to test")
    parser.add_argument("--wait", type=float, default=3.0,
                        help="Seconds to wait after each zoom for observation")
    args = parser.parse_args()

    cfg = load_config("config/config.yaml")
    logger = setup_logger("app", "INFO", "logs/test_zoom.log")

    # Init ISAPI
    isapi = HikISAPI(cfg.hik.ip, cfg.hik.user, cfg.hik.password, cfg.hik.channel)

    # Query resolutions
    print("\n--- Resolution Query ---")
    native = isapi.get_native_resolution()
    streaming = isapi.get_streaming_resolution()
    print(f"  Native resolution: {native}")
    print(f"  Streaming resolution: {streaming}")

    # Use streaming resolution as frame size (matches RTSP)
    frame_w, frame_h = streaming or native or (1920, 1080)
    print(f"  Using frame size: {frame_w}x{frame_h}")

    # Init SDK + PTZ
    sdk = HikSDK(cfg.hik.sdk_lib_dir)
    ptz = HikPTZ(sdk, cfg.hik.ip, cfg.hik.port,
                  cfg.hik.user, cfg.hik.password, cfg.hik.channel,
                  isapi=isapi)

    ptz.login()
    ptz.check_3d_positioning()

    # Test bbox: center-ish area of the frame
    test_bbox = (
        frame_w // 4, frame_h // 4,
        frame_w // 4 * 3, frame_h // 4 * 3,
    )

    try:
        if args.backend in ("sdk", "both"):
            test_sdk_zoom(ptz, *test_bbox, frame_w, frame_h)
            print(f"\n  Waiting {args.wait}s to observe camera...")
            time.sleep(args.wait)
            # Go back to preset 1
            print("  Returning to preset 1...")
            ptz.goto_preset(1)
            time.sleep(2)

        if args.backend in ("isapi", "both"):
            native_res = native or (frame_w, frame_h)
            # Scale bbox to native resolution
            sx = native_res[0] / frame_w
            sy = native_res[1] / frame_h
            scaled = (
                int(test_bbox[0] * sx), int(test_bbox[1] * sy),
                int(test_bbox[2] * sx), int(test_bbox[3] * sy),
            )
            test_isapi_zoom(isapi, *scaled, native_res[0], native_res[1])
            print(f"\n  Waiting {args.wait}s to observe camera...")
            time.sleep(args.wait)
            print("  Returning to preset 1...")
            ptz.goto_preset(1)
            time.sleep(2)

    finally:
        ptz.logout()
        print("\nDone.")


if __name__ == "__main__":
    main()
