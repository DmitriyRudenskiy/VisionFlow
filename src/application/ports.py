# src/application/ports.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from src.domain.pipeline.entities import PipelineAggregate
from src.domain.image.entities import ImageFile
from src.domain.metadata.entities import ColorPalette, PoseData


class PipelineRepository(ABC):
    """Интерфейс репозитория для сохранения состояния пайплайна"""

    @abstractmethod
    def save(self, pipeline: PipelineAggregate) -> None: ...

    @abstractmethod
    def find_by_id(self, id: UUID) -> Optional[PipelineAggregate]: ...


class StoragePort(ABC):
    """Порт для файловых операций и персистентности данных"""

    @abstractmethod
    def scan_directory(self, path: Path, recursive: bool = True) -> list[Path]: ...

    @abstractmethod
    def move_file(self, source: Path, destination: Path) -> None: ...

    @abstractmethod
    def copy_file(self, source: Path, destination: Path) -> None: ...

    @abstractmethod
    def create_directory(self, path: Path) -> None: ...

    @abstractmethod
    def get_file_hash(self, path: Path, algorithm: str = "md5") -> str: ...

    @abstractmethod
    def get_file_size(self, path: Path) -> int: ...

    @abstractmethod
    def get_file_modified_time(self, path: Path) -> float: ...

    @abstractmethod
    def persist_text(self, path: Path, content: str, encoding: str = "utf-8") -> None: ...

    @abstractmethod
    def load_text(self, path: Path, encoding: str = "utf-8") -> str: ...

    @abstractmethod
    def persist_json(self, path: Path, data: Any, encoding: str = "utf-8") -> None: ...

    @abstractmethod
    def load_json(self, path: Path, encoding: str = "utf-8") -> Any: ...

    @abstractmethod
    def path_exists(self, path: Path) -> bool: ...


class VisualDuplicateDetectorPort(ABC):
    """Порт для поиска визуальных дубликатов (pHash + ViT)"""

    @abstractmethod
    def calculate_phash(self, image_path: Path) -> str: ...

    @abstractmethod
    def calculate_vit_similarity(
        self, image_path1: Path, image_path2: Path
    ) -> float: ...


class ImageSegmentationPort(ABC):
    """Порт для AI-сегментации и кропа (SAM3)"""

    @abstractmethod
    def crop_image(self, image_path: Path, mode: str = "square") -> Path: ...


class ImageEmbeddingExtractorPort(ABC):
    """Порт для извлечения эмбеддингов (Qwen-VL)"""

    @abstractmethod
    def get_embedding(self, image_path: Path) -> list[float]: ...


class PoseExtractionPort(ABC):
    """Порт для извлечения скелета позы (DWPose)"""

    @abstractmethod
    def extract_keypoints(self, image_path: Path) -> dict: ...


class ColorPaletteExtractorPort(ABC):
    """Порт для извлечения палитры цветов"""

    @abstractmethod
    def extract_palette(self, image_path: Path, num_colors: int = 5) -> list[dict]: ...


class ContentSafetyClassifierPort(ABC):
    """Порт для классификации NSFW контента"""

    @abstractmethod
    def classify(self, image_path: Path) -> tuple[float, float]: ...


# --- Зарезервировано для будущего слоя персистентности доменных сущностей ---
class ImageRepository(ABC):
    @abstractmethod
    def save(self, image: ImageFile) -> None: ...

    @abstractmethod
    def find_by_path(self, path: Path) -> Optional[ImageFile]: ...


class MetadataRepository(ABC):
    @abstractmethod
    def save_palette(self, palette: ColorPalette) -> None: ...

    @abstractmethod
    def save_pose(self, pose: PoseData) -> None: ...