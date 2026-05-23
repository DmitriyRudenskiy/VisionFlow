from typing import Dict, List
from src.application.pipeline.steps.base_step import BaseStep
from src.domain.pipeline.exceptions import StepNotFoundError


class StepRegistry:
    """Реестр шагов пайплайна"""

    def __init__(self) -> None:
        self._steps: Dict[int, BaseStep] = {}

    def register(self, step_number: int, step: BaseStep) -> None:
        self._steps[step_number] = step

    def get(self, step_number: int) -> BaseStep:
        if step_number not in self._steps:
            raise StepNotFoundError(f"Step {step_number} not found in registry")
        return self._steps[step_number]

    def get_step_numbers(self) -> List[int]:
        """Возвращает отсортированный список номеров зарегистрированных шагов."""
        return sorted(self._steps.keys())