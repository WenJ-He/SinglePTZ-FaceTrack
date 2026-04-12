"""Display backend abstraction: web (MJPEG) or opencv mode."""

import logging
import select
import sys
from typing import Optional

import cv2
import numpy as np

from src.ui.web_stream import MjpegStreamServer

logger = logging.getLogger("app")


class DisplayBackend:
    """Unified display interface, switching between web and opencv."""

    def __init__(self, mode: str = "web", **kwargs):
        self.mode = mode
        if mode == "web":
            self._web = MjpegStreamServer(
                host=kwargs.get("web_host", "0.0.0.0"),
                port=kwargs.get("web_port", 8080),
                jpeg_quality=kwargs.get("jpeg_quality", 70),
            )
            self._web.start()
        else:
            self._web = None

    def show(self, annotated_frame: np.ndarray):
        """Display annotated frame."""
        if self._web is not None:
            self._web.push_frame(annotated_frame)
        else:
            cv2.imshow("SinglePTZ-FaceTrack", annotated_frame)
            cv2.waitKey(1)

    def poll_command(self) -> Optional[str]:
        """Poll for commands from web API or stdin."""
        # Check web commands
        if self._web is not None:
            cmd = self._web.poll_command()
            if cmd:
                return cmd

        # Check stdin (non-blocking)
        return self._poll_stdin()

    def _poll_stdin(self) -> Optional[str]:
        """Non-blocking stdin read for terminal commands."""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().lower()
                if line:
                    return line
        except Exception:
            pass
        return None

    def close(self):
        """Cleanup."""
        if self.mode != "web":
            cv2.destroyAllWindows()
