from abc import ABC, abstractmethod
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO

class BaseStep(ABC):
    """Базовый интерфейс для всех шагов пайплайна (Use Cases)"""
    @abstractmethod
    def execute(self, config: StepConfigDTO) -> StepResultDTO: ...