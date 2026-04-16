"""RTSP video source for the refactored runtime."""

import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("app")


class RtspVideoSource:
    """Background RTSP reader that always exposes the newest frame."""

    def __init__(self, url: str, reconnect_interval: float = 2.0, max_retries: int = 12):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._ts: float = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("RTSP source started: %s", self.url)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    @property
    def timestamp(self) -> float:
        with self._lock:
            return self._ts

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("RTSP source stopped")

    def _loop(self):
        retries = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                retries += 1
                logger.warning("RTSP open failed, retry %s/%s", retries, self.max_retries)
                if retries > self.max_retries:
                    logger.error("RTSP max retries exceeded")
                    break
                self._stop.wait(self.reconnect_interval)
                continue

            retries = 0
            logger.info("RTSP stream connected")
            while not self._stop.is_set():
                ok = cap.grab()
                if not ok:
                    break
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    with self._lock:
                        self._latest = frame
                        self._ts = time.time()
            cap.release()
            if not self._stop.is_set():
                logger.warning("RTSP stream disconnected, reconnecting...")
                self._stop.wait(self.reconnect_interval)
