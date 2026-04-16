"""Detection models used by the refactored runtime."""

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from src.geometry import is_edge_bbox

logger = logging.getLogger("app")

PERSON_CLASS_ID = 0


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    cls_id: int = 0


class _BaseYoloDetector:
    def __init__(self, onnx_path: str, input_size: int, conf: float, iou: float, providers, edge_reject_enabled: bool, edge_margin: int):
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.edge_reject_enabled = edge_reject_enabled
        self.edge_margin = edge_margin
        self.providers = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=self.providers)
        self.input_name = self.sess.get_inputs()[0].name

    def _letterbox(self, img: np.ndarray):
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


class FaceDetector(_BaseYoloDetector):
    def __init__(
        self,
        onnx_path: str,
        input_size: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        providers=None,
        edge_reject_enabled: bool = False,
        edge_margin: int = 5,
    ):
        super().__init__(onnx_path, input_size, conf, iou, providers, edge_reject_enabled, edge_margin)
        logger.info(
            "FaceDetector loaded: %s size=%s conf=%.2f providers=%s",
            onnx_path,
            input_size,
            conf,
            self.sess.get_providers(),
        )

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        h, w = img_bgr.shape[:2]
        padded, scale, pad = self._letterbox(img_bgr)
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        raw = self.sess.run(None, {self.input_name: blob})[0]
        preds = raw[0].T
        preds = preds[preds[:, 4] >= self.conf]
        if preds.shape[0] == 0:
            return []

        boxes = preds[:, :4].copy()
        boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2
        boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2
        boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2
        boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2
        scores = preds[:, 4]
        indices = cv2.dnn.NMSBoxes(boxes.astype(np.float32).tolist(), scores.tolist(), self.conf, self.iou)
        if len(indices) == 0:
            return []

        results = []
        for idx in indices.flatten():
            x1, y1, x2, y2 = boxes[idx]
            x1 = max(0, int(round((x1 - pad[0]) / scale)))
            y1 = max(0, int(round((y1 - pad[1]) / scale)))
            x2 = min(w, int(round((x2 - pad[0]) / scale)))
            y2 = min(h, int(round((y2 - pad[1]) / scale)))
            bbox = (x1, y1, x2, y2)
            if self.edge_reject_enabled and is_edge_bbox(bbox, w, h, self.edge_margin):
                continue
            results.append(Detection(bbox=bbox, score=float(scores[idx]), cls_id=0))
        return results


class PersonDetector(_BaseYoloDetector):
    def __init__(
        self,
        onnx_path: str,
        input_size: int = 640,
        conf: float = 0.45,
        iou: float = 0.5,
        providers=None,
        edge_reject_enabled: bool = False,
        edge_margin: int = 5,
    ):
        super().__init__(onnx_path, input_size, conf, iou, providers, edge_reject_enabled, edge_margin)
        logger.info(
            "PersonDetector loaded: %s size=%s conf=%.2f providers=%s",
            onnx_path,
            input_size,
            conf,
            self.sess.get_providers(),
        )

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        h, w = img_bgr.shape[:2]
        padded, scale, pad = self._letterbox(img_bgr)
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        raw = self.sess.run(None, {self.input_name: blob})[0]
        preds = raw[0].T
        person_scores = preds[:, 4 + PERSON_CLASS_ID]
        mask = person_scores >= self.conf
        preds = preds[mask]
        person_scores = person_scores[mask]
        if preds.shape[0] == 0:
            return []

        boxes = preds[:, :4].copy()
        boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2
        boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2
        boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2
        boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2
        indices = cv2.dnn.NMSBoxes(boxes.astype(np.float32).tolist(), person_scores.tolist(), self.conf, self.iou)
        if len(indices) == 0:
            return []

        results = []
        for idx in indices.flatten():
            x1, y1, x2, y2 = boxes[idx]
            x1 = max(0, int(round((x1 - pad[0]) / scale)))
            y1 = max(0, int(round((y1 - pad[1]) / scale)))
            x2 = min(w, int(round((x2 - pad[0]) / scale)))
            y2 = min(h, int(round((y2 - pad[1]) / scale)))
            bbox = (x1, y1, x2, y2)
            if self.edge_reject_enabled and is_edge_bbox(bbox, w, h, self.edge_margin):
                continue
            results.append(Detection(bbox=bbox, score=float(person_scores[idx]), cls_id=PERSON_CLASS_ID))
        return results
