# src/infrastructure/ai/qwen_client.py
from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

from src.application.ports import ImageEmbeddingExtractorPort

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


class QwenVLClient(ImageEmbeddingExtractorPort):
    """Адаптер для извлечения эмбеддингов через Qwen-VL."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        logger.info(f"Loading Qwen-VL model from {model_path} ...")
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" if self._device != "cpu" else None,
        )
        if self._device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        logger.info("Qwen-VL model loaded.")

    def get_embedding(self, image_path: Path) -> list[float]:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}],
            }
        ]
        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state.mean(dim=1)

        vector = embedding.squeeze().float().cpu().numpy().tolist()
        if not vector:
            raise ValueError("Empty embedding received from model")
        return vector