from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any

from src.shared.base import BaseValueObject


class PipelineStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StepConfig(BaseValueObject):
    """Конфигурация конкретного шага пайплайна (immutable)"""

    params: Dict[str, Any] = field(default_factory=dict)