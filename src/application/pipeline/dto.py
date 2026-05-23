# application/pipeline/dto.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal


@dataclass
class PipelineConfigDTO:
    """Конфигурация для запуска всего пайплайна."""
    source_path: Path
    output_path: Path
    steps_to_run: Optional[List[int]] = None
    stop_on_error: bool = True


@dataclass
class StepConfigDTO:
    """Конфигурация для конкретного шага."""
    step_number: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResultDTO:
    """Результат выполнения шага."""
    step_number: int
    status: Literal["COMPLETED", "FAILED", "SKIPPED"]
    message: str = ""
    processed_count: int = 0
    skipped_count: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ImageProcessingResultDTO:
    source_path: Path
    processed_path: Optional[Path] = None
    is_duplicate: bool = False
    error: Optional[str] = None


@dataclass
class MetadataResultDTO:
    file_path: Path
    vector: Optional[List[float]] = None
    colors: Optional[List[Dict]] = None
    nsfw_score: Optional[float] = None
    pose_keypoints: Optional[Dict] = None