from typing import Dict, List
from src.application.pipeline.steps.base_step import BaseStep
from src.domain.pipeline.exceptions import StepNotFoundError


class StepRegistry:
    """Реестр шагов пайплайна"""

    def __init__(self) -> None:
        self._steps: Dict[int, BaseStep] = {}

    def register(self, sequence_number: int, step: BaseStep) -> None:
        self._steps[sequence_number] = step

    def get(self, sequence_number: int) -> BaseStep:
        if sequence_number not in self._steps:
            raise StepNotFoundError(f"Step {sequence_number} not found in registry")
        return self._steps[sequence_number]

    def get_registered_sequence_numbers(self) -> List[int]:
        """Возвращает отсортированный список номеров зарегистрированных шагов."""
        return sorted(self._steps.keys())