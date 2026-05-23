import math
from dataclasses import dataclass
from typing import List, Tuple

from src.shared.base import BaseValueObject
from src.domain.metadata.exceptions import InvalidScoreRange


@dataclass(frozen=True)
class VectorEmbedding(BaseValueObject):
    """Векторное представление изображения (эмбеддинг)"""
    values: Tuple[float, ...]  # Tuple для неизменяемости
    model_name: str

    @property
    def dimension(self) -> int:
        return len(self.values)

    def l2_normalize(self) -> 'VectorEmbedding':
        """Возвращает новый VectorEmbedding с L2-нормализацией"""
        norm = math.sqrt(sum(v ** 2 for v in self.values))
        if norm == 0:
            return self  # Вектор из нулей останется нулевым
        normalized = tuple(v / norm for v in self.values)
        return VectorEmbedding(values=normalized, model_name=self.model_name)


@dataclass(frozen=True)
class ColorEntry(BaseValueObject):
    """Запись цвета в палитре"""
    rgb: Tuple[int, int, int]
    hex: str
    percentage: float


@dataclass(frozen=True)
class NsfwScore(BaseValueObject):
    """Оценка NSFW контента"""
    nsfw_value: float  # 0.0 - 1.0
    safe_value: float  # 0.0 - 1.0

    def __post_init__(self):
        if not (0.0 <= self.nsfw_value <= 1.0) or not (0.0 <= self.safe_value <= 1.0):
            raise InvalidScoreRange("NSFW and Safe values must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PoseKeypoint(BaseValueObject):
    """Ключевая точка скелета (Keypoint)"""
    x: float
    y: float
    confidence: float
    name: str