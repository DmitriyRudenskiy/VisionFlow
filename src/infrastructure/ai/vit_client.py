# src/infrastructure/ai/vit_client.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.application.ports import VisualDuplicateDetectorPort

logger = logging.getLogger(__name__)

try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except Exception:
    IMAGEHASH_AVAILABLE = False
    logger.warning("imagehash not installed. phash will use a simple fallback.")


class VisionTransformerClient(VisualDuplicateDetectorPort):
    """Детекция визуальных дубликатов: pHash + lightweight similarity."""

    def calculate_phash(self, image_path: Path) -> str:
        if IMAGEHASH_AVAILABLE:
            return str(imagehash.phash(Image.open(image_path)))
        # Fallback: 32×32 grayscale average hash
        img = (
            Image.open(image_path)
            .convert("L")
            .resize((32, 32), Image.Resampling.LANCZOS)
        )
        arr = np.array(img, dtype=np.float32)
        avg = arr.mean()
        bits = (arr > avg).flatten()
        return "".join("1" if b else "0" for b in bits)

    def calculate_vit_similarity(self, image_path1: Path, image_path2: Path) -> float:
        """Lightweight proxy: cosine similarity между уменьшенными RGB-векторами."""

        def _vector(path: Path) -> np.ndarray:
            img = (
                Image.open(path)
                .convert("RGB")
                .resize((64, 64), Image.Resampling.LANCZOS)
            )
            arr = np.array(img, dtype=np.float32).flatten()
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr

        v1 = _vector(image_path1)
        v2 = _vector(image_path2)
        return float(np.dot(v1, v2))