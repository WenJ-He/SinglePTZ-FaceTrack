"""Hikvision ISAPI HTTP interface for high-quality JPEG capture and PTZ control."""

import logging
from typing import Optional, Tuple

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
        self._base_url = f"http://{ip}/ISAPI"
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

    def get_native_resolution(self) -> Optional[Tuple[int, int]]:
        """Query camera native resolution via ISAPI.

        Returns:
            (width, height) or None on failure.
        """
        url = f"{self._base_url}/Image/channels/{self.channel}/capabilities"
        try:
            resp = requests.get(url, auth=self._auth, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"ISAPI resolution query failed: HTTP {resp.status_code}")
                return None
            import re
            text = resp.text
            # Try to find resolution from VideoResolution or similar tags
            match = re.search(
                r'<videoResolution>\s*<width>(\d+)</width>\s*<height>(\d+)</height>',
                text,
            )
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                logger.info(f"ISAPI native resolution: {w}x{h}")
                return (w, h)
            # Fallback: try eDFSResolution or other formats
            match = re.search(r'<width>(\d+)</width>.*?<height>(\d+)</height>', text)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                logger.info(f"ISAPI resolution (fallback parse): {w}x{h}")
                return (w, h)
            logger.warning(f"ISAPI: could not parse resolution from response")
            return None
        except requests.RequestException as e:
            logger.warning(f"ISAPI resolution query error: {e}")
            return None

    def get_streaming_resolution(self) -> Optional[Tuple[int, int]]:
        """Query streaming channel resolution (what RTSP actually delivers)."""
        url = (
            f"{self._base_url}/Streaming/channels/"
            f"{self.channel}01/capabilities"
        )
        try:
            resp = requests.get(url, auth=self._auth, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            import re
            text = resp.text
            match = re.search(
                r'<videoResolutionWidth>(\d+)</videoResolutionWidth>.*?'
                r'<videoResolutionHeight>(\d+)</videoResolutionHeight>',
                text,
            )
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                logger.info(f"ISAPI streaming resolution: {w}x{h}")
                return (w, h)
            return None
        except requests.RequestException:
            return None

    def ptz_drag_zoom(self, x1: int, y1: int, x2: int, y2: int,
                      screen_w: int, screen_h: int) -> bool:
        """3D positioning via ISAPI ptzDrag.

        Simulates dragging from center of screen to target bbox center,
        causing the camera to zoom into that region.

        Args:
            x1, y1, x2, y2: target bbox in pixel coordinates
            screen_w, screen_h: native camera resolution (NOT RTSP resolution)
        """
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        screen_cx = screen_w // 2
        screen_cy = screen_h // 2

        url = f"{self._base_url}/PTZCtrl/channels/{self.channel}/ptzDrag"
        body = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<PTZDragData>'
            f'<screenSize>{screen_w},{screen_h}</screenSize>'
            f'<startPoint>{screen_cx},{screen_cy}</startPoint>'
            f'<endPoint>{cx},{cy}</endPoint>'
            f'</PTZDragData>'
        )

        logger.info(
            f"  ISAPI ptzDrag: screen={screen_w}x{screen_h}, "
            f"start=({screen_cx},{screen_cy}), end=({cx},{cy})"
        )

        try:
            resp = requests.put(
                url,
                data=body.encode("utf-8"),
                auth=self._auth,
                timeout=self.timeout,
                headers={"Content-Type": "application/xml"},
            )
            if resp.status_code in (200, 201):
                logger.info(f"  ISAPI ptzDrag OK: HTTP {resp.status_code}")
                return True
            else:
                logger.warning(
                    f"  ISAPI ptzDrag failed: HTTP {resp.status_code}, "
                    f"body={resp.text[:200]}"
                )
                return False
        except requests.RequestException as e:
            logger.warning(f"  ISAPI ptzDrag error: {e}")
            return False

    def get_ptz_status(self) -> Optional[dict]:
        """Get current PTZ absolute position.

        Returns dict with keys: azimuth, elevation, absoluteZoom.
        azimuth/elevation in degrees, absoluteZoom as multiplier (1.0=wide).
        """
        url = f"{self._base_url}/PTZCtrl/channels/{self.channel}/absoluteEx"
        try:
            resp = requests.get(url, auth=self._auth, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            import re
            az = re.search(r'<azimuth>([^<]+)</azimuth>', resp.text)
            el = re.search(r'<elevation>([^<]+)</elevation>', resp.text)
            zm = re.search(r'<absoluteZoom>([^<]+)</absoluteZoom>', resp.text)
            if az and el and zm:
                return {
                    "azimuth": float(az.group(1)),
                    "elevation": float(el.group(1)),
                    "absoluteZoom": float(zm.group(1)),
                }
            return None
        except requests.RequestException:
            return None

    def ptz_absolute_zoom(self, azimuth: float, elevation: float,
                          absolute_zoom: float) -> bool:
        """Move PTZ to absolute position via ISAPI.

        azimuth: horizontal angle in degrees
        elevation: vertical angle in degrees
        absolute_zoom: zoom multiplier (1.0=wide, max depends on camera)
        """
        url = f"{self._base_url}/PTZCtrl/channels/{self.channel}/absoluteEx"
        body = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<PTZAbsoluteEx>'
            f'<elevation>{elevation:.2f}</elevation>'
            f'<azimuth>{azimuth:.2f}</azimuth>'
            f'<absoluteZoom>{absolute_zoom:.2f}</absoluteZoom>'
            f'</PTZAbsoluteEx>'
        )
        logger.info(
            f"  ISAPI absolute: az={azimuth:.2f}, el={elevation:.2f}, "
            f"zoom={absolute_zoom:.2f}"
        )
        try:
            resp = requests.put(
                url,
                data=body.encode("utf-8"),
                auth=self._auth,
                timeout=self.timeout,
                headers={"Content-Type": "application/xml"},
            )
            if resp.status_code in (200, 201):
                logger.info(f"  ISAPI absolute OK: HTTP {resp.status_code}")
                return True
            else:
                logger.warning(
                    f"  ISAPI absolute failed: HTTP {resp.status_code}, "
                    f"body={resp.text[:200]}"
                )
                return False
        except requests.RequestException as e:
            logger.warning(f"  ISAPI absolute error: {e}")
            return False
