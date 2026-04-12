"""YOLOv8n person detection (COCO class 0 = person)."""

import logging
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.detect.yolo_face import Detection

logger = logging.getLogger("app")

# COCO class 0 = person
PERSON_CLASS_ID = 0


class YoloPerson:
    """YOLOv8n COCO detector, filtering for person class only.

    Output shape: (1, 84, N) where 84 = 4 bbox + 80 class scores.
    """

    def __init__(self, onnx_path: str, input_size: int = 640,
                 conf: float = 0.3, iou: float = 0.5,
                 providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.input_size = input_size
        self.conf = conf
        self.iou = iou

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        logger.info(
            f"YoloPerson loaded: {onnx_path}, size={input_size}, conf={conf}"
        )

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        """Detect persons in BGR image. Returns list of Detection."""
        h, w = img_bgr.shape[:2]

        # Letterbox
        padded, scale, pad = self._letterbox(img_bgr)

        # Preprocess
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        # Inference
        outputs = self.sess.run(None, {self.input_name: blob})
        raw = outputs[0]  # (1, 84, N)

        # Transpose to (N, 84)
        preds = raw[0].T  # (N, 84)

        # Extract person class scores (index 4 + PERSON_CLASS_ID = 4)
        person_scores = preds[:, 4 + PERSON_CLASS_ID]

        # Confidence filter
        mask = person_scores >= self.conf
        preds = preds[mask]
        person_scores = person_scores[mask]
        if preds.shape[0] == 0:
            return []

        # cxcywh -> xyxy
        boxes = preds[:, :4].copy()
        boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2
        boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2
        boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2
        boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2

        # NMS
        boxes_for_nms = boxes.astype(np.float32).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, person_scores.tolist(), self.conf, self.iou,
        )
        if len(indices) == 0:
            return []

        results = []
        for idx in indices.flatten():
            x1, y1, x2, y2 = boxes[idx]
            x1 = (x1 - pad[0]) / scale
            y1 = (y1 - pad[1]) / scale
            x2 = (x2 - pad[0]) / scale
            y2 = (y2 - pad[1]) / scale
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(w, int(round(x2)))
            y2 = min(h, int(round(y2)))
            results.append(Detection(
                (x1, y1, x2, y2), float(person_scores[idx]),
                cls_id=PERSON_CLASS_ID,
            ))

        return results

    def _letterbox(self, img: np.ndarray):
        """Resize with letterbox."""
        h, w = img.shape[:2]
        target = self.input_size

        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target - new_w
        pad_h = target - new_h
        top = pad_h // 2
        left = pad_w // 2

        padded = np.full((target, target, 3), 114, dtype=np.uint8)
        padded[top:top + new_h, left:left + new_w] = resized

        return padded, scale, (left, top)
