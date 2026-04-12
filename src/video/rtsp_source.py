"""RTSP video source with background thread frame caching."""

import threading
import time
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("app")


class RtspSource:
    """Background thread grabs frames, main thread reads latest frame.

    Drops old frames to keep the newest one.
    Auto-reconnects on stream failure.
    """

    def __init__(self, url: str, reconnect_interval: float = 2.0,
                 max_retries: int = 12):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background capture thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"RTSP source started: {self.url}")

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame (or None if not available)."""
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    @property
    def timestamp(self) -> float:
        with self._lock:
            return self._ts

    def stop(self) -> None:
        """Stop background capture thread."""
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
                logger.warning(
                    f"RTSP open failed, retry {retries}/{self.max_retries}"
                )
                if retries > self.max_retries:
                    logger.error("RTSP max retries exceeded, giving up")
                    break
                self._stop.wait(self.reconnect_interval)
                continue

            logger.info("RTSP stream connected")
            retries = 0

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
