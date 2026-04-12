"""Face gallery: build from photo/ directory, cache to NPZ, match against query."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.recognize.arcface import ArcFace
from src.detect.yolo_face import YoloFace

logger = logging.getLogger("app")


@dataclass
class GalleryEntry:
    name: str
    feat: np.ndarray  # (512,) L2 normalized
    src_files: List[str] = field(default_factory=list)


@dataclass
class MatchResult:
    kind: str  # "hit" / "stranger" / "ambiguous"
    name: Optional[str]
    sim: float


class FaceGallery:
    """Face gallery built from photo/ directory.

    - Scans image files, detects face, extracts ArcFace embedding
    - Groups by filename stem (before extension)
    - Caches to NPZ for fast reload
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, arcface: ArcFace, face_det: YoloFace,
                 cache_path: str = "cache/gallery.npz"):
        self.arcface = arcface
        self.face_det = face_det
        self.cache_path = cache_path
        self.entries: Dict[str, GalleryEntry] = {}
        self._gallery_matrix: Optional[np.ndarray] = None
        self._gallery_names: List[str] = []

    def build_or_load(self, photo_dir: str = "photo") -> None:
        """Build gallery from photo dir, or load from cache if unchanged."""
        if self._try_load_cache(photo_dir):
            logger.info(f"Gallery loaded from cache: {len(self.entries)} entries")
            return

        logger.info(f"Building gallery from {photo_dir}...")
        self.entries.clear()

        # Group files by identity name
        groups: Dict[str, List[str]] = {}
        for fname in sorted(os.listdir(photo_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in self.SUPPORTED_EXTS:
                continue
            stem = os.path.splitext(fname)[0]
            # Handle suffixes like "姓名_01.jpg" -> "姓名"
            name = stem.split("_")[0] if "_" in stem else stem
            path = os.path.join(photo_dir, fname)
            groups.setdefault(name, []).append(path)

        # Process each group
        for name, files in groups.items():
            feats = []
            for fpath in files:
                feat = self._process_file(fpath)
                if feat is not None:
                    feats.append(feat)

            if not feats:
                logger.warning(f"No face found for '{name}', skipping")
                continue

            # Average embeddings if multiple images
            mean_feat = np.mean(feats, axis=0)
            mean_feat /= (np.linalg.norm(mean_feat) + 1e-9)
            self.entries[name] = GalleryEntry(
                name=name, feat=mean_feat, src_files=files,
            )

        self._rebuild_matrix()
        self._save_cache(photo_dir)
        logger.info(f"Gallery built: {len(self.entries)} identities")

    def match(self, feat: np.ndarray,
              match_th: float = 0.35,
              reject_th: float = 0.20) -> MatchResult:
        """Match a query embedding against the gallery.

        Returns:
            hit: sim >= match_th
            stranger: sim < reject_th
            ambiguous: in between
        """
        if self._gallery_matrix is None or len(self._gallery_names) == 0:
            return MatchResult("stranger", None, 0.0)

        sims = self._gallery_matrix @ feat  # (K,)
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])
        top_name = self._gallery_names[top_idx]

        if top_sim >= match_th:
            return MatchResult("hit", top_name, top_sim)
        if top_sim < reject_th:
            return MatchResult("stranger", None, top_sim)
        return MatchResult("ambiguous", top_name, top_sim)

    def _process_file(self, path: str) -> Optional[np.ndarray]:
        """Detect face in image and return embedding."""
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Cannot read image: {path}")
            return None

        dets = self.face_det.detect(img)
        if not dets:
            logger.warning(f"No face detected in: {path}")
            return None

        # Pick largest face
        best = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        x1, y1, x2, y2 = best.bbox
        # Expand slightly for better alignment
        h, w = img.shape[:2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        fw, fh = (x2 - x1) * 1.3, (y2 - y1) * 1.3
        x1e = max(int(cx - fw / 2), 0)
        y1e = max(int(cy - fh / 2), 0)
        x2e = min(int(cx + fw / 2), w)
        y2e = min(int(cy + fh / 2), h)

        crop = img[y1e:y2e, x1e:x2e]
        if crop.size == 0:
            return None

        return self.arcface.embed(crop)

    def _rebuild_matrix(self):
        """Rebuild gallery matrix for fast matching."""
        if not self.entries:
            self._gallery_matrix = None
            self._gallery_names = []
            return

        self._gallery_names = list(self.entries.keys())
        self._gallery_matrix = np.stack(
            [self.entries[n].feat for n in self._gallery_names]
        )

    def _try_load_cache(self, photo_dir: str) -> bool:
        """Try loading from cache if source files unchanged."""
        if not os.path.isfile(self.cache_path):
            return False

        try:
            data = np.load(self.cache_path, allow_pickle=True)
            cached_mtimes = dict(data.get("mtimes", []))

            # Check if any source file changed
            for fname in os.listdir(photo_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTS:
                    continue
                path = os.path.join(photo_dir, fname)
                mt = os.path.getmtime(path)
                if cached_mtimes.get(fname) != mt:
                    return False

            # Load entries
            names = list(data["names"])
            feats = data["feats"]
            files_list = data.get("files", [[] for _ in names])

            for i, name in enumerate(names):
                self.entries[name] = GalleryEntry(
                    name=name,
                    feat=feats[i],
                    src_files=list(files_list[i]) if i < len(files_list) else [],
                )
            self._rebuild_matrix()
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}, rebuilding")
            return False

    def _save_cache(self, photo_dir: str):
        """Save gallery to NPZ cache."""
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)

        names = list(self.entries.keys())
        feats = np.stack([self.entries[n].feat for n in names])
        files_arr = [self.entries[n].src_files for n in names]

        # Record mtimes
        mtimes = []
        for fname in os.listdir(photo_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in self.SUPPORTED_EXTS:
                continue
            path = os.path.join(photo_dir, fname)
            mtimes.append((fname, os.path.getmtime(path)))

        np.savez(
            self.cache_path,
            names=names,
            feats=feats,
            files=np.array(files_arr, dtype=object),
            mtimes=np.array(mtimes, dtype=object),
        )
        logger.info(f"Gallery cached to {self.cache_path}")
