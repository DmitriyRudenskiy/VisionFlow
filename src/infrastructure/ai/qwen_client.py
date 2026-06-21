# src/infrastructure/ai/qwen_client.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
from PIL import Image

from src.application.ports import ImageEmbeddingExtractorPort

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


class QwenVLClient(ImageEmbeddingExtractorPort):
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "auto") -> None:
        self._model_path = model_path
        self._device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"

        logger.info(f"Loading Qwen-VL model on {self._device}...")
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            device_map="auto" if self._device == "cuda" else None
        )

        if self._device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        logger.info("Qwen-VL model loaded.")

    def get_embedding(self, image_path: Path) -> List[float]:
        image = Image.open(image_path).convert("RGB")

        # Формирование промпта (для VL моделей часто нужен пустой текстовый промпт для получения эмбеддинга изображения)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        text_prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self._processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Обычно pooler_output - это эмбеддинг всего изображения/промпта
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                # Fallback: усреднение токенов
                embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.squeeze().float().cpu().numpy().tolist()