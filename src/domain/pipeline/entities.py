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

    sequence_number: int
    name: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: StepStatus = StepStatus.PENDING
    config: StepConfig = field(default_factory=StepConfig)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def update_modified_timestamp(self) -> None:
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
                f"Cannot start step '{self.name}' in status {self.status.value}. "
                f"Allowed: PENDING, FAILED."
            )
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.error = None
        self.update_modified_timestamp()

    def complete(self) -> None:
        if self.status != StepStatus.RUNNING:
            raise InvalidStepStateTransition(
                f"Cannot complete step '{self.name}' in status {self.status.value}. Must be RUNNING."
            )
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.update_modified_timestamp()

    def fail(self, error: str) -> None:
        if self.status != StepStatus.RUNNING:
            raise InvalidStepStateTransition(
                f"Cannot fail step '{self.name}' in status {self.status.value}. Must be RUNNING."
            )
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        self.update_modified_timestamp()


@dataclass
class PipelineAggregate:
    """Агрегат корня пайплайна"""

    name: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PipelineStatus = PipelineStatus.PENDING
    steps: List[PipelineStep] = field(default_factory=list)
    source_directory: Path = Path()
    output_directory: Path = Path()

    def update_modified_timestamp(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipelineAggregate):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def add_step(self, step: PipelineStep) -> None:
        existing_index = next(
            (i for i, s in enumerate(self.steps) if s.sequence_number == step.sequence_number),
            None,
        )
        if existing_index is not None:
            self.steps[existing_index] = step
        else:
            self.steps.append(step)
        self.update_modified_timestamp()

    def find_step(self, sequence_number: int) -> Optional[PipelineStep]:
        for step in self.steps:
            if step.sequence_number == sequence_number:
                return step
        return None

    def _get_step(self, sequence_number: int) -> PipelineStep:
        step = self.find_step(sequence_number)
        if step is None:
            raise StepNotFoundError(f"Step with number {sequence_number} not found")
        return step

    def resume(self) -> None:
        if self.status == PipelineStatus.FAILED:
            self.status = PipelineStatus.PENDING
        for step in self.steps:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.FAILED
                step.error = "Pipeline was interrupted during execution"
                step.completed_at = datetime.now(timezone.utc)
                step.update_modified_timestamp()
        self.update_modified_timestamp()

    def start_step(self, sequence_number: int) -> None:
        if self.status == PipelineStatus.COMPLETED:
            raise InvalidStepStateTransition("Pipeline is already completed")
        if self.status not in {PipelineStatus.PENDING, PipelineStatus.RUNNING}:
            raise InvalidStepStateTransition(
                f"Cannot start step in {self.status.value} state"
            )
        if self.status == PipelineStatus.PENDING:
            self.status = PipelineStatus.RUNNING
        step = self._get_step(sequence_number)
        step.start()
        self.update_modified_timestamp()

    def complete_step(self, sequence_number: int) -> None:
        step = self._get_step(sequence_number)
        step.complete()
        if all(s.status == StepStatus.COMPLETED for s in self.steps):
            self.status = PipelineStatus.COMPLETED
        self.update_modified_timestamp()

    def fail_step(self, sequence_number: int, error: str, critical: bool = True) -> None:
        step = self._get_step(sequence_number)
        step.fail(error)
        if critical:
            self.status = PipelineStatus.FAILED
        self.update_modified_timestamp()