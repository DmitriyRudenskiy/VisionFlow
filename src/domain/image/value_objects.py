# domain/image/value_objects.py
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.shared.base import BaseValueObject
from src.domain.image.exceptions import InvalidImageFormat

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"})


@dataclass(frozen=True)
class FilePath(BaseValueObject):
    """Value Object для пути к файлу. Отвечает за валидацию формата пути и расширения."""
    path: Path

    def __post_init__(self):
        if not isinstance(self.path, Path):
            object.__setattr__(self, 'path', Path(self.path))
        if not self.path.suffix:
            raise ValueError(f"Path '{self.path}' has no extension")
        if self.path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidImageFormat(
                f"Unsupported file extension '{self.path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class ImageMetadata(BaseValueObject):
    """Метаданные файла (immutable)"""
    original_name: str
    extension: str
    size_bytes: int
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None