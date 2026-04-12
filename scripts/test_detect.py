"""Detection visualization test: RTSP -> face1280 + person -> draw boxes -> display."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from src.config import load_config, auto_providers
from src.utils.logger import setup_logger
from src.video.rtsp_source import RtspSource
from src.detect.yolo_face import YoloFace
from src.detect.yolo_person import YoloPerson


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    setup_logger("app", cfg.log.level, cfg.log.file)

    providers = auto_providers(cfg.runtime.prefer_gpu)

    # Init detectors
    face_det = YoloFace(cfg.models.face_wide, input_size=1280,
                        conf=cfg.detect.face_wide_conf,
                        iou=cfg.detect.face_wide_iou,
                        providers=providers)
    person_det = YoloPerson(cfg.models.person, input_size=640,
                            conf=cfg.detect.person_conf,
                            iou=cfg.detect.person_iou,
                            providers=providers)

    # Start RTSP
    src = RtspSource(cfg.hik.rtsp_url)
    src.start()

    print("Waiting for frames... Press q to quit.")
    fps_ts = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            frame = src.read()
            if frame is None:
                time.sleep(0.05)
                continue

            # Face detection
            faces = face_det.detect(frame)
            # Person detection
            persons = person_det.detect(frame)

            # Draw face boxes (green)
            for det in faces:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"face {det.score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw person boxes (blue)
            for det in persons:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, f"person {det.score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            # FPS
            frame_count += 1
            if time.time() - fps_ts >= 1.0:
                fps = frame_count / (time.time() - fps_ts)
                frame_count = 0
                fps_ts = time.time()

            cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(faces)} | Persons: {len(persons)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
