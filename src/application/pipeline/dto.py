# src/application/pipeline/dto.py
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

    sequence_number: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResultDTO:
    """Результат выполнения шага."""

    sequence_number: int
    status: Literal["COMPLETED", "FAILED", "SKIPPED"]
    message: str = ""
    processed_count: int = 0
    skipped_count: int = 0
    errors: Optional[List[str]] = field(default=None)

    @classmethod
    def completed(
        cls,
        sequence_number: int,
        message: str = "",
        processed_count: int = 0,
        skipped_count: int = 0,
        errors: Optional[List[str]] = None,
    ) -> "StepResultDTO":
        return cls(
            sequence_number=sequence_number,
            status="COMPLETED",
            message=message,
            processed_count=processed_count,
            skipped_count=skipped_count,
            errors=errors,
        )

    @classmethod
    def failed(
        cls,
        sequence_number: int,
        message: str = "",
        errors: Optional[List[str]] = None,
    ) -> "StepResultDTO":
        return cls(
            sequence_number=sequence_number,
            status="FAILED",
            message=message,
            errors=errors or [],
        )

    @classmethod
    def skipped(cls, sequence_number: int, message: str = "Step already completed.") -> "StepResultDTO":
        return cls(
            sequence_number=sequence_number,
            status="SKIPPED",
            message=message,
        )