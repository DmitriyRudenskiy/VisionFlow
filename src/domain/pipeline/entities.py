# domain/pipeline/entities.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

from src.domain.pipeline.value_objects import PipelineStatus, StepStatus, StepConfig
from src.domain.pipeline.exceptions import InvalidStepStateTransition, StepNotFoundError


@dataclass
class PipelineStep:
    """Сущность шага пайплайна"""

    step_number: int
    step_name: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: StepStatus = StepStatus.PENDING
    config: StepConfig = field(default_factory=StepConfig)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipelineStep):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def start(self) -> None:
        if self.status not in {StepStatus.PENDING, StepStatus.FAILED}:
            raise InvalidStepStateTransition(
                f"Cannot start step '{self.step_name}' in status {self.status.value}. "
                f"Allowed: PENDING, FAILED."
            )
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.error = None
        self.touch()

    def complete(self) -> None:
        if self.status != StepStatus.RUNNING:
            raise InvalidStepStateTransition(
                f"Cannot complete step '{self.step_name}' in status {self.status.value}. Must be RUNNING."
            )
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.touch()

    def fail(self, error: str) -> None:
        if self.status != StepStatus.RUNNING:
            raise InvalidStepStateTransition(
                f"Cannot fail step '{self.step_name}' in status {self.status.value}. Must be RUNNING."
            )
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        self.touch()


@dataclass
class PipelineAggregate:
    """Агрегат корня пайплайна"""

    name: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PipelineStatus = PipelineStatus.PENDING
    steps: List[PipelineStep] = field(default_factory=list)
    source_path: Path = Path()
    output_path: Path = Path()

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipelineAggregate):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def add_step(self, step: PipelineStep) -> None:
        self.steps.append(step)
        self.touch()

    def _get_step(self, step_number: int) -> PipelineStep:
        for step in self.steps:
            if step.step_number == step_number:
                return step
        raise StepNotFoundError(f"Step with number {step_number} not found")

    def resume(self) -> None:
        """Позволяет продолжить выполнение после FAILED или аварийного прерывания."""
        if self.status == PipelineStatus.FAILED:
            self.status = PipelineStatus.PENDING
        # Сброс зависших шагов (например, после аварийного завершения процесса)
        for step in self.steps:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.FAILED
                step.error = "Pipeline was interrupted during execution"
                step.completed_at = datetime.now(timezone.utc)
                step.touch()
        self.touch()

    def start_step(self, step_number: int) -> None:
        if self.status == PipelineStatus.COMPLETED:
            raise InvalidStepStateTransition("Pipeline is already completed")
        if self.status not in {PipelineStatus.PENDING, PipelineStatus.RUNNING}:
            raise InvalidStepStateTransition(
                f"Cannot start step in {self.status.value} state"
            )
        if self.status == PipelineStatus.PENDING:
            self.status = PipelineStatus.RUNNING
        step = self._get_step(step_number)
        step.start()
        self.touch()

    def complete_step(self, step_number: int) -> None:
        step = self._get_step(step_number)
        step.complete()
        if all(s.status == StepStatus.COMPLETED for s in self.steps):
            self.status = PipelineStatus.COMPLETED
        self.touch()

    def fail_step(self, step_number: int, error: str, critical: bool = True) -> None:
        step = self._get_step(step_number)
        step.fail(error)
        if critical:
            self.status = PipelineStatus.FAILED
        self.touch()
