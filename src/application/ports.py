from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
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


class FileSystemServicePort(ABC):
    """Порт для файловых операций"""

    @abstractmethod
    def scan_directory(self, path: Path, recursive: bool = True) -> List[Path]: ...

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


class VisualDupDetectorPort(ABC):
    """Порт для поиска визуальных дубликатов (pHash + ViT)"""

    @abstractmethod
    def calculate_phash(self, image_path: Path) -> str: ...

    @abstractmethod
    def calculate_vit_similarity(self, image_path1: Path, image_path2: Path) -> float: ...


class AISegmenterPort(ABC):
    """Порт для AI-сегментации и кропа (SAM3)"""

    @abstractmethod
    def crop_image(self, image_path: Path, mode: str = "square") -> Path: ...


class VectorizationPort(ABC):
    """Порт для извлечения эмбеддингов (Qwen-VL)"""

    @abstractmethod
    def get_embedding(self, image_path: Path) -> List[float]: ...


class PoseExtractorPort(ABC):
    """Порт для извлечения скелета позы (DWPose)"""

    @abstractmethod
    def extract_keypoints(self, image_path: Path) -> dict: ...


class ColorExtractorPort(ABC):
    """Порт для извлечения палитры цветов"""

    @abstractmethod
    def extract_palette(self, image_path: Path, num_colors: int = 5) -> List[dict]: ...


class NsfwClassifierPort(ABC):
    """Порт для классификации NSFW контента"""

    @abstractmethod
    def classify(self, image_path: Path) -> Tuple[float, float]: ...


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