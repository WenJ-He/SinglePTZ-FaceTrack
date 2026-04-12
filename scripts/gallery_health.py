"""Gallery health check: N x N cosine similarity matrix + heatmap."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, auto_providers
from src.utils.logger import setup_logger
from src.recognize.arcface import ArcFace
from src.detect.yolo_face import YoloFace
from src.recognize.gallery import FaceGallery


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    setup_logger("app", cfg.log.level, cfg.log.file)

    providers = auto_providers(cfg.runtime.prefer_gpu)

    arcface = ArcFace(cfg.models.arcface, providers=providers)
    face_det = YoloFace(cfg.models.face_close, input_size=640,
                        conf=cfg.detect.face_close_conf,
                        providers=providers)

    gallery = FaceGallery(arcface, face_det, "cache/gallery.npz")
    gallery.build_or_load("photo")

    names = gallery._gallery_names
    matrix = gallery._gallery_matrix  # (N, 512)
    N = len(names)

    print(f"Gallery: {N} identities")

    # Compute NxN cosine similarity
    sim_matrix = matrix @ matrix.T  # (N, N)

    # Print statistics
    same_sims = []
    cross_sims = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            val = sim_matrix[i, j]
            if i < j:  # avoid double counting
                cross_sims.append(val)

    print(f"\n=== Gallery Health Report ===")
    print(f"Identities: {N}")

    # Self-similarity (should be 1.0 by construction)
    for i in range(N):
        assert abs(sim_matrix[i, i] - 1.0) < 1e-6, f"Self-sim != 1.0 for {names[i]}"

    # Cross-person statistics
    if cross_sims:
        cross_arr = np.array(cross_sims)
        print(f"Cross-person similarity: min={cross_arr.min():.4f}, "
              f"max={cross_arr.max():.4f}, mean={cross_arr.mean():.4f}")

        # Flag suspicious pairs
        threshold = 0.3
        flagged = []
        for i in range(N):
            for j in range(i + 1, N):
                if sim_matrix[i, j] > threshold:
                    flagged.append((names[i], names[j], sim_matrix[i, j]))

        if flagged:
            print(f"\nWARNING: {len(flagged)} pairs with cross-similarity > {threshold}:")
            for a, b, s in sorted(flagged, key=lambda x: -x[2]):
                print(f"  {a} vs {b}: {s:.4f}")
        else:
            print(f"\nNo suspicious pairs (cross-sim > {threshold})")

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(12, N * 0.3), max(10, N * 0.3)))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Face Gallery Similarity Matrix")
    plt.tight_layout()

    output_path = "cache/gallery_heatmap.png"
    os.makedirs("cache", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nHeatmap saved to: {output_path}")


if __name__ == "__main__":
    main()
