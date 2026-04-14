"""YOLO face detection (640 & 1280 input sizes)."""

import logging
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.geometry import is_edge_bbox

logger = logging.getLogger("app")


class Detection:
    """Single detection result."""
    __slots__ = ("bbox", "score", "cls_id")

    def __init__(self, bbox: Tuple[int, int, int, int],
                 score: float, cls_id: int = 0):
        self.bbox = bbox  # x1, y1, x2, y2
        self.score = score
        self.cls_id = cls_id

    def __repr__(self):
        return f"Det(bbox={self.bbox}, score={self.score:.3f})"


class YoloFace:
    """YOLO11n-face detector supporting 640 and 1280 input sizes.

    Output shape: (1, 5, N) where 5 = cx, cy, w, h, conf.
    Single class (face), no built-in NMS.
    """

    def __init__(self, onnx_path: str, input_size: int = 640,
                 conf: float = 0.35, iou: float = 0.5,
                 providers=None,
                 edge_reject_enabled: bool = False,
                 edge_margin: int = 5):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.edge_reject_enabled = edge_reject_enabled
        self.edge_margin = edge_margin

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        input_shape = self.sess.get_inputs()[0].shape
        logger.info(
            f"YoloFace loaded: {onnx_path}, input={input_shape}, "
            f"size={input_size}, conf={conf}, "
            f"edge_reject={edge_reject_enabled}, edge_margin={edge_margin}"
        )

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        """Detect faces in BGR image. Returns list of Detection."""
        h, w = img_bgr.shape[:2]

        # Letterbox resize
        padded, scale, pad = self._letterbox(img_bgr)

        # Preprocess: BGR -> RGB, normalize, NCHW
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        # Inference
        outputs = self.sess.run(None, {self.input_name: blob})
        raw = outputs[0]  # (1, 5, N)

        # Transpose to (N, 5): cx, cy, w, h, conf
        preds = raw[0].T  # (N, 5)

        # Confidence filter
        mask = preds[:, 4] >= self.conf
        preds = preds[mask]
        if preds.shape[0] == 0:
            return []

        # Convert cxcywh -> xyxy
        boxes = preds[:, :4].copy()
        boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2  # x1
        boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2  # y1
        boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2  # x2
        boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2  # y2
        scores = preds[:, 4]

        # NMS
        boxes_for_nms = boxes.astype(np.float32).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, scores.tolist(), self.conf, self.iou,
        )
        if len(indices) == 0:
            return []

        # Map back to original image coordinates
        results = []
        for idx in indices.flatten():
            x1, y1, x2, y2 = boxes[idx]
            # Reverse letterbox
            x1 = (x1 - pad[0]) / scale
            y1 = (y1 - pad[1]) / scale
            x2 = (x2 - pad[0]) / scale
            y2 = (y2 - pad[1]) / scale
            # Clip to frame
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(w, int(round(x2)))
            y2 = min(h, int(round(y2)))
            bbox = (x1, y1, x2, y2)
            # Edge rejection
            if self.edge_reject_enabled and is_edge_bbox(
                    bbox, w, h, self.edge_margin):
                continue
            results.append(Detection(bbox, float(scores[idx])))

        return results

    def _letterbox(self, img: np.ndarray):
        """Resize with letterbox (keep aspect ratio, pad with gray)."""
        h, w = img.shape[:2]
        target = self.input_size

        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad
        pad_w = target - new_w
        pad_h = target - new_h
        top = pad_h // 2
        left = pad_w // 2

        padded = np.full((target, target, 3), 114, dtype=np.uint8)
        padded[top:top + new_h, left:left + new_w] = resized

        return padded, scale, (left, top)
