"""ArcFace face embedding module."""

import logging
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("app")


class ArcFace:
    """ArcFace face feature extractor.

    Input: BGR face crop (any size) -> resize 112x112 -> embed -> L2 normalize
    Output: (512,) L2-normalized embedding vector
    """

    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        logger.info(f"ArcFace loaded: {onnx_path}")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 512-d embedding from a face crop."""
        img = cv2.resize(face_bgr, (112, 112))
        img = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 112, 112)

        feat = self.sess.run(None, {self.input_name: img})[0][0]  # (512,)
        norm = np.linalg.norm(feat)
        if norm < 1e-9:
            return feat
        return feat / norm

    def embed_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for multiple faces. Returns (N, 512)."""
        if not faces:
            return np.empty((0, 512), dtype=np.float32)

        blobs = []
        for face in faces:
            img = cv2.resize(face, (112, 112))
            img = img[:, :, ::-1].astype(np.float32)
            img = (img - 127.5) / 128.0
            img = img.transpose(2, 0, 1)
            blobs.append(img)

        batch = np.stack(blobs)  # (N, 3, 112, 112)
        feats = self.sess.run(None, {self.input_name: batch})[0]  # (N, 512)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        return feats / norms
