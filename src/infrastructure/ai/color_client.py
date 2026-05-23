# src/infrastructure/ai/color_client.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.application.ports import ColorPaletteExtractorPort

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans  # type: ignore[import-untyped]
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False
    logger.warning("sklearn not installed, using cv2 fallback for K-Means")

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False


class ColorExtractorClient(ColorPaletteExtractorPort):
    """Извлечение палитры через KMeans кластеризацию."""

    MAX_PIXELS = 500_000

    def extract_palette(self, image_path: Path, num_colors: int = 20) -> list[dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Cannot open image {image_path}: {e}")
            raise RuntimeError(f"Failed to open image {image_path}: {e}") from e

        width, height = image.size
        total_pixels = width * height
        if total_pixels > self.MAX_PIXELS:
            ratio = (self.MAX_PIXELS / total_pixels) ** 0.5
            new_w = max(1, int(width * ratio))
            new_h = max(1, int(height * ratio))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.debug(f"Resized {width}×{height} → {new_w}×{new_h} for color extraction")

        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        if _SKLEARN_AVAILABLE:
            kmeans = KMeans(
                n_clusters=num_colors,
                random_state=42,
                n_init=10,
                max_iter=300,
                algorithm="lloyd",
            )
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
        elif _CV2_AVAILABLE:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.2)
            best_labels = np.empty((pixels.shape[0], 1), dtype=np.int32)
            _, labels, centers = cv2.kmeans(
                pixels, num_colors, best_labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            colors = centers.astype(int)
            labels = labels.flatten()
        else:
            raise RuntimeError("Neither sklearn nor cv2 is installed.")

        counts = np.bincount(labels, minlength=num_colors)
        total = len(labels)

        palette = []
        for idx, count in enumerate(counts):
            if count == 0:
                continue

            rgb = colors[idx].tolist()
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            palette.append({
                "rgb": rgb,
                "hex": hex_color,
                "percentage": round(count / total * 100, 2),
            })

        palette.sort(key=lambda c: c["percentage"], reverse=True)
        return palette