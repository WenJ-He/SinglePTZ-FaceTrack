"""OSNet ReID feature extraction module."""

import logging

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("app")


class OSNetReID:
    """OSNet_x0_25 person re-identification feature extractor.

    Input: BGR person crop (any size) -> resize(128,256) -> ImageNet normalize -> 512-d
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        logger.info(f"OSNetReID loaded: {onnx_path}")

    def embed(self, person_bgr: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 512-d embedding from a person crop."""
        img = cv2.resize(person_bgr, (128, 256))
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 256, 128)

        feat = self.sess.run(None, {self.input_name: img})[0][0]  # (512,)
        norm = np.linalg.norm(feat)
        if norm < 1e-9:
            return feat
        return feat / norm
