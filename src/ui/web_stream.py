"""MJPEG HTTP stream server for web-based visualization."""

import logging
import queue
import threading
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

logger = logging.getLogger("app")


class MjpegStreamServer:
    """MJPEG HTTP stream server with command interface."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080,
                 jpeg_quality: int = 70):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._frame_queue = queue.Queue(maxsize=2)
        self._command_queue = queue.Queue()
        self._app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self._app.route("/")
        def index():
            return (
                '<html><head><title>SinglePTZ-FaceTrack</title></head>'
                '<body style="margin:0;background:#000">'
                '<img src="/stream" style="width:100%;height:auto">'
                '<div style="position:fixed;bottom:10px;left:10px;color:#aaa">'
                'Commands: q=quit r=reset h=home p=pause v=record<br>'
                'POST /api/command {"action":"quit|reset|home|pause|record"}'
                '</div></body></html>'
            )

        @self._app.route("/stream")
        def stream():
            def generate():
                while True:
                    try:
                        frame_bytes = self._frame_queue.get(timeout=5.0)
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n"
                               + frame_bytes + b"\r\n")
                    except queue.Empty:
                        continue
            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self._app.route("/api/command", methods=["POST"])
        def command():
            action = request.json.get("action", "") if request.json else ""
            if action:
                self._command_queue.put(action)
            return jsonify({"ok": True, "action": action})

    def start(self):
        """Start Flask in background thread."""
        t = threading.Thread(
            target=self._app.run,
            kwargs={"host": self.host, "port": self.port, "threaded": True},
            daemon=True,
        )
        t.start()
        logger.info(f"MJPEG stream at http://{self.host}:{self.port}/")

    def push_frame(self, annotated_frame: np.ndarray):
        """Encode and push frame to MJPEG stream."""
        _, buf = cv2.imencode(
            ".jpg", annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        try:
            self._frame_queue.put_nowait(buf.tobytes())
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(buf.tobytes())
            except queue.Full:
                pass

    def poll_command(self) -> Optional[str]:
        """Non-blocking command poll."""
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None
