# application/pipeline/orchestrator.py
from typing import Generator, List, Optional
from uuid import UUID
import logging

from src.application.pipeline.dto import PipelineConfigDTO, StepConfigDTO, StepResultDTO
from src.application.pipeline.step_registry import StepRegistry
from src.application.ports import PipelineRepository
from src.domain.pipeline.entities import PipelineAggregate, PipelineStep
from src.domain.pipeline.value_objects import StepStatus

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, step_registry: StepRegistry, pipeline_repo: PipelineRepository):
        self._registry = step_registry
        self._repo = pipeline_repo

    def get_available_steps(self) -> List[int]:
        return self._registry.get_step_numbers()

    def create_pipeline(self, source_path, output_path) -> PipelineAggregate:
        return PipelineAggregate(
            name=f"Pipeline-{source_path.name}",
            source_path=source_path,
            output_path=output_path
        )

    def execute(
        self,
        config: PipelineConfigDTO,
        pipeline_id: Optional[UUID] = None
    ) -> Generator[StepResultDTO, None, None]:
        if pipeline_id:
            pipeline = self._repo.find_by_id(pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline with ID {pipeline_id} not found")
            pipeline.resume()
        else:
            pipeline = self.create_pipeline(config.source_path, config.output_path)
            self._repo.save(pipeline)

        step_numbers: List[int] = config.steps_to_run or self._registry.get_step_numbers()
        step_numbers = sorted(step_numbers)

        existing_numbers = {s.step_number for s in pipeline.steps}
        for step_num in step_numbers:
            if step_num not in existing_numbers:
                step_instance = self._registry.get(step_num)
                step_entity = PipelineStep(step_number=step_num, step_name=step_instance.__class__.__name__)
                pipeline.add_step(step_entity)
                existing_numbers.add(step_num)
        self._repo.save(pipeline)

        for step_num in step_numbers:
            existing_step = next((s for s in pipeline.steps if s.step_number == step_num), None)
            if existing_step and existing_step.status == StepStatus.COMPLETED:
                logger.info(f"Step {step_num} already completed, skipping.")
                continue

            step_instance = self._registry.get(step_num)

            step_config = StepConfigDTO(
                step_number=step_num,
                params={
                    "source_path": str(config.source_path),
                    "output_path": str(config.output_path)
                }
            )

            pipeline.start_step(step_num)
            self._repo.save(pipeline)
            logger.info(f"Starting step {step_num}: {step_instance.__class__.__name__}")

            try:
                result = step_instance.execute(step_config)

                if result.status == "FAILED":
                    pipeline.fail_step(step_num, result.message or "Step returned FAILED status", critical=config.stop_on_error)
                    self._repo.save(pipeline)
                    yield result
                    if config.stop_on_error:
                        return
                else:
                    pipeline.complete_step(step_num)
                    self._repo.save(pipeline)
                    yield result

            except Exception as e:
                logger.exception(f"Unhandled exception in step {step_num}")
                pipeline.fail_step(step_num, str(e), critical=config.stop_on_error)
                self._repo.save(pipeline)

                yield StepResultDTO(
                    step_number=step_num,
                    status="FAILED",
                    message=str(e),
                    errors=[str(e)]
                )
                if config.stop_on_error:
                    return