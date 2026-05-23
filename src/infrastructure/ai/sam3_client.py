# src/infrastructure/ai/sam3_client.py
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.application.ports import ImageSegmentationPort

logger = logging.getLogger(__name__)

_sam3_import_error: Exception | None = None

try:
    from ultralytics.models.sam import SAM3SemanticPredictor  # type: ignore[import-untyped]
    _SAM3_AVAILABLE = True
except Exception as _exc:
    _sam3_import_error = _exc
    SAM3SemanticPredictor = None  # type: ignore[misc,assignment]
    _SAM3_AVAILABLE = False


class SAM3Client(ImageSegmentationPort):
    """Адаптер для SAM3: сегментация по текстовому промпту и кроп объекта в квадрат."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        if not _SAM3_AVAILABLE or SAM3SemanticPredictor is None:
            raise RuntimeError(
                "ultralytics is not installed. Install it to use SAM3Client."
            ) from _sam3_import_error

        self._device = self._resolve_device(device)
        self._predictor = self._load_model(model_path)
        logger.info("SAM3 model loaded.")

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str != "auto":
            return device_str
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self, checkpoint_path: str):
        overrides = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": 644,
            "save": False,
            "half": False,
            "verbose": False,
            "model": checkpoint_path,
            "device": self._device,
        }
        try:
            return SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}") from e

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def _crop_to_square(
        self,
        image: Image.Image,
        mask: np.ndarray,
        mode: str,
    ) -> Image.Image:
        use_mask = mode in ("mask", "transparent")
        transparent = mode == "transparent"

        bbox = self._mask_bbox(mask)
        if bbox is None:
            return image
        xmin, ymin, xmax, ymax = bbox

        # padding 0% by default (можно вынести в параметр при необходимости)
        pad_w = int((xmax - xmin) * 0.0)
        pad_h = int((ymax - ymin) * 0.0)
        xmin -= pad_w
        ymin -= pad_h
        xmax += pad_w
        ymax += pad_h

        rect_w = xmax - xmin
        rect_h = ymax - ymin
        square_size = max(rect_w, rect_h)

        if transparent:
            result = Image.new("RGBA", (square_size, square_size), (0, 0, 0, 0))
        else:
            result = Image.new("RGB", (square_size, square_size), (247, 247, 247))

        img_w, img_h = image.size
        src_xmin = max(0, xmin)
        src_ymin = max(0, ymin)
        src_xmax = min(img_w, xmax)
        src_ymax = min(img_h, ymax)

        if src_xmin >= src_xmax or src_ymin >= src_ymax:
            return result

        cx = square_size // 2
        cy = square_size // 2
        ideal_cx = (xmin + xmax) // 2
        ideal_cy = (ymin + ymax) // 2
        paste_x = cx - (ideal_cx - src_xmin)
        paste_y = cy - (ideal_cy - src_ymin)

        cropped = image.crop((src_xmin, src_ymin, src_xmax, src_ymax))
        if transparent and cropped.mode != "RGBA":
            cropped = cropped.convert("RGBA")

        if use_mask:
            mask_crop = mask[src_ymin:src_ymax, src_xmin:src_xmax]
            mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8), mode="L")
            result.paste(cropped, (paste_x, paste_y), mask_pil)
        else:
            result.paste(cropped, (paste_x, paste_y))

        return result

    def crop_image(self, image_path: Path, mode: str = "square") -> Path:
        if mode not in ("square", "mask", "transparent"):
            raise ValueError(f"Unsupported crop mode: {mode}")

        image: Image.Image = Image.open(image_path)
        if mode == "transparent" and image.mode != "RGBA":
            image = image.convert("RGBA")

        # --- Детекция ---
        masks: list[np.ndarray] = []
        prompts = ["person"]
        fallback = ["object", "thing", "item", "woman", "people", "character"]

        for prompt in prompts + fallback:
            try:
                self._predictor.set_image(str(image_path))
                results = self._predictor(text=[prompt])
            except Exception as e:
                logger.warning(f"SAM3 prediction failed for prompt '{prompt}': {e}")
                continue

            if results and results[0].masks is not None:
                for mask_data in results[0].masks.data:
                    mask = mask_data.cpu().numpy().astype(bool)
                    if np.any(mask):
                        masks.append(mask)
                if masks:
                    logger.info(f"SAM3 found objects with prompt '{prompt}'")
                    break

        if not masks:
            logger.warning(f"No objects found in {image_path}, returning original")
            return image_path

        # Берём самый крупный объект по площади bbox
        def _area(m: np.ndarray) -> int:
            bb = self._mask_bbox(m)
            return 0 if bb is None else (bb[2] - bb[0]) * (bb[3] - bb[1])

        best_mask = max(masks, key=_area)
        result_img = self._crop_to_square(image, best_mask, mode)

        suffix = ".png" if mode == "transparent" else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        save_kwargs: dict = {}
        if suffix == ".jpg":
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0

        result_img.save(str(tmp_path), **save_kwargs)
        return tmp_path