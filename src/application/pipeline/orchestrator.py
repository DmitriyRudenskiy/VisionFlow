from pathlib import Path
from typing import Dict, Generator, List, Optional, Set
from uuid import UUID
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading

from src.application.pipeline.dto import PipelineConfigDTO, StepConfigDTO, StepResultDTO
from src.application.pipeline.step_registry import StepRegistry
from src.application.ports import PipelineRepository
from src.domain.pipeline.entities import PipelineAggregate, PipelineStep
from src.domain.pipeline.value_objects import StepStatus, PipelineStatus

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    _PARALLEL_STEP_GROUPS: List[Set[int]] = [{5, 6, 7, 8}]

    def __init__(self, step_registry: StepRegistry, pipeline_repo: PipelineRepository):
        self._registry = step_registry
        self._repo = pipeline_repo
        self._save_lock = threading.Lock()

    def get_registered_sequence_numbers(self) -> List[int]:
        return self._registry.get_registered_sequence_numbers()

    def create_pipeline(self, source_directory: Path, output_directory: Path) -> PipelineAggregate:
        return PipelineAggregate(
            name=f"Pipeline-{source_directory.name}",
            source_directory=source_directory,
            output_directory=output_directory,
        )

    def _save_pipeline(self, pipeline: PipelineAggregate) -> None:
        with self._save_lock:
            self._repo.save(pipeline)

    def _build_step_configuration(self, sequence_number: int, config: PipelineConfigDTO) -> StepConfigDTO:
        return StepConfigDTO(
            sequence_number=sequence_number,
            params={
                "source_path": str(config.source_path),
                "output_path": str(config.output_path),
            },
        )

    def _attempt_step_activation(
        self, pipeline: PipelineAggregate, sequence_number: int
    ) -> Optional[StepResultDTO]:
        try:
            pipeline.start_step(sequence_number)
            self._save_pipeline(pipeline)
            return None
        except Exception as e:
            logger.error(f"Cannot start step {sequence_number}: {e}")
            return StepResultDTO.failed(
                sequence_number=sequence_number,
                message=f"Failed to start step: {e}",
                errors=[str(e)],
            )

    def _commit_step_result(
        self,
        pipeline: PipelineAggregate,
        sequence_number: int,
        result: StepResultDTO,
        config: PipelineConfigDTO,
    ) -> StepResultDTO:
        if result.status == "FAILED":
            try:
                pipeline.fail_step(
                    sequence_number,
                    result.message or "Step returned FAILED status",
                    critical=config.halt_on_failure,
                )
                self._save_pipeline(pipeline)
            except Exception as e:
                logger.error(f"Could not mark step {sequence_number} as failed: {e}")
                self._save_pipeline(pipeline)
        else:
            try:
                pipeline.complete_step(sequence_number)
                self._save_pipeline(pipeline)
            except Exception as e:
                logger.error(f"Could not mark step {sequence_number} as completed: {e}")
                self._save_pipeline(pipeline)
        return result

    def _execute_sequential_step(
        self, pipeline: PipelineAggregate, sequence_number: int, config: PipelineConfigDTO
    ) -> StepResultDTO:
        start_error = self._attempt_step_activation(pipeline, sequence_number)
        if start_error:
            return start_error

        step_instance = self._registry.get(sequence_number)
        step_config = self._build_step_configuration(sequence_number, config)
        logger.info(f"Executing step {sequence_number}: {step_instance.__class__.__name__}")

        try:
            result = step_instance.execute(step_config)
        except Exception as e:
            logger.exception(f"Unhandled exception in step {sequence_number}")
            result = StepResultDTO.failed(
                sequence_number=sequence_number, message=str(e), errors=[str(e)]
            )

        return self._commit_step_result(pipeline, sequence_number, result, config)

    def _execute_concurrent_group(
        self,
        pipeline: PipelineAggregate,
        group_steps: List[int],
        config: PipelineConfigDTO,
    ) -> Generator[StepResultDTO, None, None]:
        pending_steps: List[int] = []

        for sequence_number in group_steps:
            existing_step = pipeline.find_step(sequence_number)
            if existing_step and existing_step.status == StepStatus.COMPLETED:
                yield StepResultDTO.skipped(sequence_number)
                continue

            start_error = self._attempt_step_activation(pipeline, sequence_number)
            if start_error:
                yield start_error
                if config.halt_on_failure:
                    return
                continue

            pending_steps.append(sequence_number)

        if not pending_steps:
            return

        future_to_step: Dict[Future, int] = {}
        with ThreadPoolExecutor(
            max_workers=len(pending_steps), thread_name_prefix="parallel_step"
        ) as executor:
            for sequence_number in pending_steps:
                step_instance = self._registry.get(sequence_number)
                step_config = self._build_step_configuration(sequence_number, config)
                future = executor.submit(step_instance.execute, step_config)
                future_to_step[future] = sequence_number

            stop_yielding = False
            for future in as_completed(future_to_step):
                sequence_number = future_to_step[future]

                if future.cancelled():
                    result = StepResultDTO.failed(
                        sequence_number=sequence_number,
                        message="Step cancelled due to failure of another step in parallel group.",
                    )
                else:
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.exception(f"Unhandled exception in step {sequence_number}")
                        result = StepResultDTO.failed(
                            sequence_number=sequence_number, message=str(e), errors=[str(e)]
                        )

                if stop_yielding:
                    self._commit_step_result(pipeline, sequence_number, result, config)
                    continue

                yield self._commit_step_result(pipeline, sequence_number, result, config)

                if result.status == "FAILED" and config.halt_on_failure:
                    stop_yielding = True
                    for f in future_to_step:
                        if not f.done():
                            f.cancel()

    def _synchronize_registered_steps(self, pipeline: PipelineAggregate, step_numbers: List[int]) -> None:
        existing_numbers = {s.sequence_number for s in pipeline.steps}
        available_registry = set(self._registry.get_registered_sequence_numbers())

        for step_num in step_numbers:
            if step_num in existing_numbers:
                continue
            if step_num not in available_registry:
                logger.warning(f"Step {step_num} requested but not found in registry.")
                continue

            step_instance = self._registry.get(step_num)
            step_entity = PipelineStep(
                sequence_number=step_num,
                name=step_instance.__class__.__name__,
            )
            pipeline.add_step(step_entity)

    def execute(
        self, config: PipelineConfigDTO, pipeline_id: Optional[UUID] = None
    ) -> Generator[StepResultDTO, None, None]:
        if pipeline_id:
            pipeline = self._repo.find_by_id(pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline with ID {pipeline_id} not found")
            pipeline.resume()
        else:
            pipeline = self.create_pipeline(config.source_path, config.output_path)
            self._save_pipeline(pipeline)

        if config.steps_to_run is not None:
            step_numbers = sorted(config.steps_to_run)
        elif pipeline_id:
            step_numbers = sorted([s.sequence_number for s in pipeline.steps])
        else:
            step_numbers = self._registry.get_registered_sequence_numbers()

        self._synchronize_registered_steps(pipeline, step_numbers)
        self._save_pipeline(pipeline)

        existing_step_numbers = {s.sequence_number for s in pipeline.steps}
        completed_parallel: Set[int] = set()

        for step_num in step_numbers:
            if step_num not in existing_step_numbers:
                continue
            if step_num in completed_parallel:
                continue

            group = next((g for g in self._PARALLEL_STEP_GROUPS if step_num in g), None)
            if group:
                group_steps = [s for s in step_numbers if s in group and s not in completed_parallel]
                yield from self._execute_concurrent_group(pipeline, group_steps, config)
                completed_parallel.update(group_steps)
                if pipeline.status == PipelineStatus.FAILED and config.halt_on_failure:
                    return
            else:
                existing_step = pipeline.find_step(step_num)
                if existing_step and existing_step.status == StepStatus.COMPLETED:
                    logger.info(f"Step {step_num} already completed, skipping.")
                    yield StepResultDTO.skipped(step_num)
                    continue

                result = self._execute_sequential_step(pipeline, step_num, config)
                yield result
                if result.status == "FAILED" and config.halt_on_failure:
                    return