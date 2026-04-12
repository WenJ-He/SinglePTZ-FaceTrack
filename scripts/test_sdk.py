"""SDK smoke test: login → ability check → preset tour → zoom → record → logout."""

import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.utils.logger import setup_logger
from src.sdk.hik_sdk import HikSDK
from src.sdk.hik_ptz import HikPTZ


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    setup_logger("app", cfg.log.level, cfg.log.file)

    sdk = HikSDK(cfg.hik.sdk_lib_dir)
    ptz = HikPTZ(sdk, cfg.hik.ip, cfg.hik.port,
                  cfg.hik.user, cfg.hik.password, cfg.hik.channel)

    # Login
    print("[1/6] Logging in...")
    ptz.login()

    # Check 3D positioning
    print("[2/6] Checking 3D positioning ability...")
    has_3d = ptz.check_3d_positioning()
    if not has_3d:
        print("WARNING: 3D positioning not supported!")
    else:
        print("  3D positioning supported.")

    # Tour presets
    for preset_id in cfg.patrol.presets:
        print(f"[3/6] Going to preset {preset_id}...")
        ok = ptz.goto_preset(preset_id)
        if ok:
            print(f"  Preset {preset_id} OK, waiting 2s...")
            time.sleep(2)
        else:
            print(f"  Preset {preset_id} FAILED")

    # Zoom to center
    print("[4/6] Zoom to center of 2560x1440 frame...")
    center_bbox = (640, 360, 1920, 1080)
    ok = ptz.zoom_to_bbox(center_bbox, 2560, 1440, expand=cfg.ptz.expand_ratio)
    if ok:
        print("  Zoom OK, waiting 2s...")
        time.sleep(2)
        # Return to preset 1
        ptz.goto_preset(1)
        time.sleep(2)
    else:
        print("  Zoom FAILED")

    # Record test
    print("[5/6] Testing record start/stop...")
    ok = ptz.start_record()
    if ok:
        print("  Record started, waiting 2s...")
        time.sleep(2)
        ptz.stop_record()
        print("  Record stopped.")
    else:
        print("  Record start FAILED (may be expected if no storage configured)")

    # Logout
    print("[6/6] Logging out...")
    ptz.logout()
    print("All tests completed.")


if __name__ == "__main__":
    main()
