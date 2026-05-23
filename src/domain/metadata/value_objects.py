# domain/metadata/value_objects.py
import math
from dataclasses import dataclass
from typing import Tuple

from src.shared.base import BaseValueObject
from src.domain.metadata.exceptions import InvalidScoreRange


@dataclass(frozen=True)
class VectorEmbedding(BaseValueObject):
    """Векторное представление изображения (эмбеддинг)"""

    values: Tuple[float, ...]
    model_name: str

    @property
    def dimension(self) -> int:
        return len(self.values)

    def l2_normalize(self) -> "VectorEmbedding":
        """Возвращает новый VectorEmbedding с L2-нормализацией"""
        norm = math.sqrt(sum(v**2 for v in self.values))
        if norm == 0:
            return self
        normalized = tuple(v / norm for v in self.values)
        return VectorEmbedding(values=normalized, model_name=self.model_name)


@dataclass(frozen=True)
class ColorEntry(BaseValueObject):
    """Запись цвета в палитре"""

    rgb: Tuple[int, int, int]
    hex: str
    percentage: float

    def __post_init__(self):
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in self.rgb):
            raise ValueError(f"RGB values must be integers in 0..255, got {self.rgb}")
        if not (0.0 <= self.percentage <= 100.0):
            raise ValueError(f"Percentage must be in 0..100, got {self.percentage}")
        if (
            not isinstance(self.hex, str)
            or not self.hex.startswith("#")
            or len(self.hex) not in (4, 7)
        ):
            raise ValueError(f"Invalid hex format: {self.hex}")


@dataclass(frozen=True)
class NsfwScore(BaseValueObject):
    """Оценка NSFW контента"""

    nsfw_value: float
    safe_value: float

    def __post_init__(self):
        if not (0.0 <= self.nsfw_value <= 1.0) or not (0.0 <= self.safe_value <= 1.0):
            raise InvalidScoreRange("NSFW and Safe values must be between 0.0 and 1.0")
        total = self.nsfw_value + self.safe_value
        if not (0.99 <= total <= 1.01):
            raise InvalidScoreRange(f"NSFW + Safe must equal ~1.0, got {total}")


@dataclass(frozen=True)
class PoseKeypoint(BaseValueObject):
    """Ключевая точка скелета (Keypoint)"""

    x: float
    y: float
    confidence: float
    name: str
