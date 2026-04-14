"""SinglePTZ-FaceTrack main entry point."""

import os
import signal
import sys
import time
import logging

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.config import load_config, auto_providers
from src.utils.logger import setup_logger
from src.sdk.hik_sdk import HikSDK
from src.sdk.hik_ptz import HikPTZ
from src.video.rtsp_source import RtspSource
from src.detect.yolo_face import YoloFace
from src.detect.yolo_person import YoloPerson
from src.recognize.arcface import ArcFace
from src.recognize.gallery import FaceGallery
from src.reid.osnet import OSNetReID
from src.scheduler.state_machine import ScanScheduler
from src.sdk.hik_isapi import HikISAPI


def main():
    cfg = load_config("config/config.yaml")
    logger = setup_logger("app", cfg.log.level, cfg.log.file)

    logger.info("=" * 60)
    logger.info("SinglePTZ-FaceTrack starting")
    logger.info("=" * 60)

    # Set LD_LIBRARY_PATH for SDK
    sdk_dir = os.path.abspath(cfg.hik.sdk_lib_dir)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if sdk_dir not in existing:
        os.environ["LD_LIBRARY_PATH"] = sdk_dir + ":" + existing
    logger.info(f"SDK lib dir: {sdk_dir}")

    # ORT providers
    providers = auto_providers(cfg.runtime.prefer_gpu)

    # Init SDK + PTZ
    sdk = HikSDK(cfg.hik.sdk_lib_dir)
    ptz = HikPTZ(sdk, cfg.hik.ip, cfg.hik.port,
                  cfg.hik.user, cfg.hik.password, cfg.hik.channel,
                  min_interval=cfg.ptz.min_wait_after_cmd)

    try:
        # Login
        ptz.login()

        # Check 3D positioning
        has_3d = ptz.check_3d_positioning()
        if not has_3d:
            logger.warning(
                "Device may not support 3D positioning (PTZZoomIn). "
                "Continuing anyway; zoom calls will fail gracefully."
            )

        # Init detectors
        face_wide = YoloFace(cfg.models.face_wide, input_size=1280,
                             conf=cfg.detect.face_wide_conf,
                             iou=cfg.detect.face_wide_iou,
                             providers=providers,
                             edge_reject_enabled=cfg.detect.edge_reject_enabled,
                             edge_margin=cfg.detect.edge_margin)
        face_close = YoloFace(cfg.models.face_close, input_size=640,
                              conf=cfg.detect.face_close_conf,
                              iou=cfg.detect.face_close_iou,
                              providers=providers)
        person_det = YoloPerson(cfg.models.person, input_size=640,
                                conf=cfg.detect.person_conf,
                                iou=cfg.detect.person_iou,
                                providers=providers)

        # Init ArcFace + gallery
        arcface = ArcFace(cfg.models.arcface, providers=providers)
        face_det_for_gallery = YoloFace(cfg.models.face_close, input_size=640,
                                        conf=cfg.detect.face_close_conf,
                                        providers=providers)
        gallery = FaceGallery(arcface, face_det_for_gallery, "cache/gallery.npz")
        gallery.build_or_load("photo")

        # Init ReID
        reid = OSNetReID(cfg.models.reid, providers=providers)

        # Start RTSP
        video = RtspSource(cfg.hik.rtsp_url)
        video.start()

        # Wait for first frame
        logger.info("Waiting for video stream...")
        for _ in range(100):
            frame = video.read()
            if frame is not None:
                logger.info(f"Video stream ready: {frame.shape}")
                break
            time.sleep(0.1)
        else:
            logger.error("No video frame received after 10s")
            video.stop()
            ptz.logout()
            return

        # Init ISAPI for high-quality capture (optional)
        isapi = None
        if cfg.hik.isapi_enabled:
            isapi = HikISAPI(cfg.hik.ip, cfg.hik.user, cfg.hik.password,
                             cfg.hik.channel)
            logger.info("ISAPI high-quality capture enabled")

        # Init scheduler
        scheduler = ScanScheduler(
            cfg, ptz, video, face_wide, face_close, person_det,
            arcface=arcface, gallery=gallery, reid=reid, isapi=isapi,
        )

        # Graceful shutdown
        def signal_handler(sig, frame_num):
            logger.info(f"Signal {sig} received, stopping...")
            scheduler.stop_flag = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Stdin reader thread for commands
        import threading
        def stdin_reader():
            while not scheduler.stop_flag:
                try:
                    line = input()
                    cmd = line.strip().lower()
                    if cmd:
                        scheduler.handle_command(cmd)
                except (EOFError, KeyboardInterrupt):
                    scheduler.stop_flag = True
                    break

        stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
        stdin_thread.start()

        # Run
        scheduler.run()

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        logger.info("Shutting down...")
        try:
            video.stop()
        except Exception:
            pass
        try:
            ptz.logout()
        except Exception:
            pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
