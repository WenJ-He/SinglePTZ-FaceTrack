"""Hikvision ISAPI HTTP interface for high-quality JPEG capture."""

import logging
from typing import Optional

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

logger = logging.getLogger("app")


class HikISAPI:
    """Hikvision ISAPI REST client for snapshot capture and PTZ control.

    Uses HTTP Digest auth over ISAPI to grab hardware-encoded JPEG frames,
    bypassing H.264 decode/re-encode artifacts from RTSP.
    """

    def __init__(self, ip: str, user: str, password: str,
                 channel: int = 1, timeout: float = 3.0):
        self.ip = ip
        self.user = user
        self.password = password
        self.channel = channel
        self.timeout = timeout
        self._auth = HTTPDigestAuth(user, password)
        self._picture_url = (
            f"http://{ip}/ISAPI/Streaming/channels/{channel}01/picture"
        )

    def capture_jpeg(self) -> Optional[np.ndarray]:
        """Capture a single high-quality JPEG frame via ISAPI.

        Returns:
            BGR numpy array of the captured frame, or None on failure.
        """
        try:
            resp = requests.get(
                self._picture_url,
                auth=self._auth,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"ISAPI capture failed: HTTP {resp.status_code}"
                )
                return None
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("ISAPI capture: JPEG decode failed")
                return None
            return frame
        except requests.RequestException as e:
            logger.warning(f"ISAPI capture error: {e}")
            return None
