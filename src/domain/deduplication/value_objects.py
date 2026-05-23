# domain/deduplication/value_objects.py
from dataclasses import dataclass
from src.shared.base import BaseValueObject
from src.domain.deduplication.exceptions import InvalidSimilarityScore


@dataclass(frozen=True)
class FileHash(BaseValueObject):
    """Value Object для хеша файла (MD5, SHA256 и т.д.)"""
    algorithm: str
    value: str

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.value}"


@dataclass(frozen=True)
class SimilarityScore(BaseValueObject):
    """Value Object для оценки сходства (0.0 - 1.0)"""
    value: float

    def __post_init__(self):
        if not (0.0 <= self.value <= 1.0):
            raise InvalidSimilarityScore(
                f"Similarity score must be between 0.0 and 1.0, got {self.value}"
            )