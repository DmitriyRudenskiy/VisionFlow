# src/infrastructure/ai/dwpose_client.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from src.application.ports import PoseExtractionPort

logger = logging.getLogger(__name__)

_dwpose_import_error: Exception | None = None

try:
    from dwpose import DwposeDetector  # type: ignore[import-untyped]
    _DWPOSE_AVAILABLE = True
except Exception as _exc:
    _dwpose_import_error = _exc
    DwposeDetector = None  # type: ignore[misc,assignment]
    _DWPOSE_AVAILABLE = False


class DWPoseClient(PoseExtractionPort):
    """Адаптер для DWPose: детекция скелета, лица и рук."""

    def __init__(self, target_size: int = 1024) -> None:
        if not _DWPOSE_AVAILABLE or DwposeDetector is None:
            raise RuntimeError(
                "dwpose is not installed. Install it to use DWPoseClient."
            ) from _dwpose_import_error

        self._target_size = target_size
        logger.info("Loading DWPose model...")
        self._model = DwposeDetector.from_pretrained_default()
        logger.info("DWPose model loaded.")

    @staticmethod
    def _pad_to_square(img: Image.Image, target_size: int) -> Image.Image:
        """Вписывает изображение в квадрат с сохранением пропорций на чёрном фоне."""
        img = img.convert("RGB")
        old_w, old_h = img.size
        ratio = min(target_size / old_w, target_size / old_h)
        new_w = int(old_w * ratio)
        new_h = int(old_h * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        new_img.paste(img, (pad_x, pad_y))
        return new_img

    def extract_keypoints(self, image_path: Path) -> dict[str, Any]:
        img = Image.open(image_path)
        img_squared = self._pad_to_square(img, self._target_size)

        # Возвращает кортеж: (rendered_image, keypoints_dict, source_image)
        _, keypoints, _ = self._model(
            img_squared,
            include_hand=True,
            include_face=True,
            include_body=True,
            image_and_json=True,
            detect_resolution=self._target_size,
        )
        # keypoints ожидается в виде dict с ключами body/face/left_hand/right_hand
        if not isinstance(keypoints, dict):
            raise RuntimeError(f"Unexpected DWPose output type: {type(keypoints)}")
        return keypoints