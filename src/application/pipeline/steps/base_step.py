# src/application/pipeline/steps/base_step.py
from abc import ABC, abstractmethod
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO


class BaseStep(ABC):
    """Базовый интерфейс для всех шагов пайплайна (Use Cases)"""

    def prepare(self) -> None:
        """Ленивая инициализация тяжёлых ресурсов (модели, GPU и т.д.).
        Вызывается оркестратором непосредственно перед execute()."""
        pass

    @abstractmethod
    def execute(self, config: StepConfigDTO) -> StepResultDTO: ...