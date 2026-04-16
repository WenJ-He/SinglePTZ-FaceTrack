"""Minimal MJPEG debug server and overlay rendering."""

import logging
import queue
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

logger = logging.getLogger("app")


class WebDebugServer:
    """Serve MJPEG frames and simple PTZ/debug commands."""

    def __init__(self, host: str = "0.0.0.0", port: int = 18060, jpeg_quality: int = 90):
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
            buttons = [
                ("left", "Left"),
                ("right", "Right"),
                ("up", "Up"),
                ("down", "Down"),
                ("zoom_in", "Zoom +"),
                ("zoom_out", "Zoom -"),
                ("stop", "Stop"),
                ("home", "Home"),
                ("pause", "Pause"),
                ("quit", "Quit"),
            ]
            controls = "".join(
                f"<button onclick=\"sendCmd('{action}')\">{label}</button>"
                for action, label in buttons
            )
            return (
                "<html><head><title>SinglePTZ Stage1</title>"
                "<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}"
                ".controls{position:fixed;left:16px;bottom:16px;display:flex;gap:8px;flex-wrap:wrap}"
                "button{padding:10px 14px;background:#222;border:1px solid #555;color:#ddd;cursor:pointer}"
                "img{width:100%;height:auto;display:block}"
                "</style></head><body>"
                "<img src='/stream'>"
                f"<div class='controls'>{controls}</div>"
                "<script>"
                "function sendCmd(action){fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});}"
                "</script></body></html>"
            )

        @self._app.route("/stream")
        def stream():
            def generate():
                while True:
                    try:
                        frame_bytes = self._frame_queue.get(timeout=5.0)
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    except queue.Empty:
                        continue

            return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self._app.route("/api/command", methods=["POST"])
        def command():
            action = request.json.get("action", "") if request.json else ""
            if action:
                self._command_queue.put(action.strip().lower())
            return jsonify({"ok": True, "action": action})

    def start(self):
        thread = threading.Thread(
            target=self._app.run,
            kwargs={"host": self.host, "port": self.port, "threaded": True},
            daemon=True,
        )
        thread.start()
        logger.info("Web debug server at http://%s:%s/", self.host, self.port)

    def show(self, frame: np.ndarray):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
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

    def poll_command(self) -> str | None:
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None


class Stage1Overlay:
    """Draw detections, target boxes, and basic runtime HUD."""

    def render(
        self,
        frame,
        state: str,
        person_dets,
        face_dets,
        tracked_person=None,
        target_kind: str = "none",
        target_bbox=None,
        debug: dict | None = None,
        draw_all: bool = True,
    ):
        annotated = frame.copy()

        if draw_all:
            for det in person_dets or []:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 140, 0), 2)
            for det in face_dets or []:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

        if tracked_person is not None:
            x1, y1, x2, y2 = tracked_person.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                "tracked-person",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        if target_bbox is not None:
            x1, y1, x2, y2 = target_bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"target:{target_kind}",
                (x1, min(y2 + 20, annotated.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        lines = [f"State: {state}", f"Target: {target_kind}"]
        if debug:
            for key in ("dx", "dy", "target_ratio", "zoom_steps", "last_action", "paused", "settling", "fps"):
                if key in debug:
                    lines.append(f"{key}: {debug[key]}")

        y = 28
        for line in lines:
            cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            y += 26

        h, w = annotated.shape[:2]
        cv2.line(annotated, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
        cv2.line(annotated, (0, h // 2), (w, h // 2), (90, 90, 90), 1)
        return annotated
