from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.application.ports import ImageSegmentationPort

logger = logging.getLogger(__name__)

# Глобальное подавление логов Ultralytics
try:
    import ultralytics

    ultralytics.utils.LOGGER.setLevel(logging.ERROR)
except ImportError:
    pass

_sam3_import_error: Optional[Exception] = None
try:
    from ultralytics.models.sam import SAM3SemanticPredictor  # type: ignore[import-untyped]

    _SAM_AVAILABLE = True
except Exception as _exc:
    _sam3_import_error = _exc
    SAM3SemanticPredictor = None  # type: ignore[misc,assignment]
    _SAM_AVAILABLE = False


@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


class SAM3Client(ImageSegmentationPort):
    DEFAULT_PROMPT = "person"
    FALLBACK_PROMPTS: List[str] = [
        "object", "thing", "item", "woman", "people", "character"
    ]
    PAD_PERCENT: float = 0.0
    BACKGROUND_COLOR: Tuple[int, int, int] = (247, 247, 247)

    def __init__(self, model_path: str, device: str = "auto") -> None:
        if not _SAM_AVAILABLE or SAM3SemanticPredictor is None:
            raise RuntimeError(
                "ultralytics is not installed or SAM3SemanticPredictor not found.") from _sam3_import_error

        self._device = self._resolve_device(device)
        logger.info(f"Loading SAM3 model from {model_path} on {self._device}...")

        overrides = {
            'conf': 0.25,
            'task': 'segment',
            'mode': 'predict',
            'imgsz': 644,
            'save': False,
            'half': False,
            'verbose': False,
            'model': model_path,
            'device': self._device
        }

        try:
            self._predictor = SAM3SemanticPredictor(overrides=overrides)
            logger.info("SAM3 model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM3 Predictor: {e}") from e

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str != "auto":
            return device_str
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def crop_image(self, image_path: Path, mode: str = "square") -> List[Path]:
        """
        Вырезает все найденные объекты и возвращает список путей к временным файлам.
        """
        if mode not in ("square", "mask", "transparent"):
            raise ValueError(f"Unsupported crop mode: {mode}")

        image = Image.open(image_path)
        image_for_model = image.convert("RGB")

        transparent = mode == "transparent"
        if transparent:
            image_source = image.convert("RGBA")
        else:
            image_source = image.convert("RGB")

        # Детекция
        masks = self._detect_with_fallback(str(image_path), image_for_model)

        if not masks:
            logger.warning(f"No objects found in {image_path.name}")
            return []

        logger.info(f"Processing {len(masks)} objects from {image_path.name}...")

        saved_paths: List[Path] = []

        # Обрабатываем каждую маску
        for i, mask in enumerate(masks):
            bbox = self._get_mask_bbox(mask)
            if not bbox:
                continue

            result_img = self._crop_to_square(
                image_source, mask, bbox,
                transparent=transparent, use_mask=(mode in ("mask", "transparent"))
            )

            # Сохраняем во временный файл
            suffix = ".png" if transparent else ".jpg"
            # Создаем уникальный временный файл для каждого объекта
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()

            save_kwargs: dict[str, Any] = {"quality": 95} if suffix == ".jpg" else {}
            result_img.save(str(temp_path), **save_kwargs)
            saved_paths.append(temp_path)

        return saved_paths

    def _detect_with_fallback(self, image_path: str, image: Image.Image) -> List[np.ndarray]:
        self._predictor.set_image(image)

        masks = self._predict_prompts([self.DEFAULT_PROMPT])
        if masks:
            return masks

        for fb_prompt in self.FALLBACK_PROMPTS:
            masks = self._predict_prompts([fb_prompt])
            if masks:
                return masks

        return []

    def _predict_prompts(self, prompts: List[str]) -> List[np.ndarray]:
        try:
            results = self._predictor(text=prompts)
            if not results or results[0].masks is None:
                return []

            found_masks = []
            for mask_data in results[0].masks.data:
                mask = mask_data.cpu().numpy().astype(bool)
                if np.any(mask):
                    found_masks.append(mask)
            return found_masks
        except Exception:
            return []

    @staticmethod
    def _get_mask_bbox(mask: np.ndarray) -> Optional[BBox]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return BBox(int(xmin), int(ymin), int(xmax), int(ymax))

    def _crop_to_square(
            self,
            image: Image.Image,
            mask: np.ndarray,
            bbox: BBox,
            transparent: bool,
            use_mask: bool
    ) -> Image.Image:
        img_w, img_h = image.size
        pad_w = int(bbox.width * self.PAD_PERCENT / 100.0)
        pad_h = int(bbox.height * self.PAD_PERCENT / 100.0)

        xmin = bbox.xmin - pad_w
        ymin = bbox.ymin - pad_h
        xmax = bbox.xmax + pad_w
        ymax = bbox.ymax + pad_h

        rect_w = xmax - xmin
        rect_h = ymax - ymin
        square_size = max(rect_w, rect_h)

        bg_color: tuple[int, ...] = (0, 0, 0, 0) if transparent else self.BACKGROUND_COLOR

        result = Image.new(image.mode, (square_size, square_size), bg_color)

        src_xmin = max(0, xmin)
        src_ymin = max(0, ymin)
        src_xmax = min(img_w, xmax)
        src_ymax = min(img_h, ymax)

        if src_xmin >= src_xmax or src_ymin >= src_ymax:
            return result

        center_x = square_size // 2
        center_y = square_size // 2
        ideal_center_x = (xmin + xmax) // 2
        ideal_center_y = (ymin + ymax) // 2

        paste_x = center_x - (ideal_center_x - src_xmin)
        paste_y = center_y - (ideal_center_y - src_ymin)

        cropped_rect = image.crop((src_xmin, src_ymin, src_xmax, src_ymax))

        if use_mask:
            mask_crop = mask[src_ymin:src_ymax, src_xmin:src_xmax]
            mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8), mode='L')
            result.paste(cropped_rect, (paste_x, paste_y), mask_pil)
        else:
            result.paste(cropped_rect, (paste_x, paste_y))

        return result