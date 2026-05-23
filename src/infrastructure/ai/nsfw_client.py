# src/infrastructure/ai/nsfw_client.py
from __future__ import annotations

import logging
from pathlib import Path

from src.application.ports import ContentSafetyClassifierPort

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


class NSFWClient(ContentSafetyClassifierPort):
    """Классификация NSFW через lightweight HF-модель (fallback к заглушке при ошибках)."""

    def __init__(self, model: str = "Falconsai/nsfw_image_detection") -> None:
        self._model_name = model
        self._pipeline = None

        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading NSFW classifier: {model}")
                self._pipeline = pipeline("image-classification", model=model)
                logger.info("NSFW classifier loaded.")
            except Exception as e:
                logger.error(f"Failed to load NSFW model: {e}")

    def classify(self, image_path: Path) -> tuple[float, float]:
        if self._pipeline is None:
            logger.warning("NSFW pipeline unavailable, returning default safe scores.")
            return 0.0, 1.0

        try:
            result = self._pipeline(str(image_path))
            # Пример: [{'label': 'nsfw', 'score': 0.9}, {'label': 'normal', 'score': 0.1}]
            scores = {item["label"].lower(): item["score"] for item in result}
            nsfw = scores.get("nsfw", scores.get("unsafe", 0.0))
            safe = scores.get("normal", scores.get("safe", 1.0 - nsfw))
            return float(nsfw), float(safe)
        except Exception as e:
            logger.error(f"NSFW classification failed for {image_path}: {e}")
            return 0.0, 1.0