from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, List, Optional, Set
from uuid import UUID

from src.application.pipeline.dto import PipelineConfigDTO, StepConfigDTO, StepResultDTO
from src.application.pipeline.step_registry import StepRegistry
from src.application.ports import PipelineRepository
from src.domain.pipeline.entities import PipelineAggregate, PipelineStep
from src.domain.pipeline.value_objects import StepStatus, PipelineStatus

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Оркестратор выполнения пайплайна.

    Управляет жизненным циклом пайплайна, последовательным и параллельным
    выполнением шагов, сохранением состояния и восстановлением после сбоев.

    Attributes:
        PARALLEL_GROUPS: Множества шагов, выполняемых параллельно.
    """

    PARALLEL_GROUPS: List[Set[int]] = [{5, 6, 7, 8}]

    def __init__(self, step_registry: StepRegistry, pipeline_repo: PipelineRepository) -> None:
        self._registry = step_registry
        self._repo = pipeline_repo
        self._save_lock = threading.Lock()

    def get_available_steps(self) -> List[int]:
        """Возвращает отсортированный список номеров доступных шагов."""
        return self._registry.get_step_numbers()

    def create_pipeline(self, source_path: Path, output_path: Path) -> PipelineAggregate:
        """Создаёт новый агрегат пайплайна."""
        return PipelineAggregate(
            name=f"Pipeline-{source_path.name}",
            source_directory=source_path,
            output_directory=output_path,
        )

    def _save_pipeline(self, pipeline: PipelineAggregate) -> None:
        """Потокобезопасное сохранение состояния пайплайна."""
        with self._save_lock:
            self._repo.save(pipeline)

    def _run_single_step(
        self, pipeline: PipelineAggregate, step_num: int, config: PipelineConfigDTO
    ) -> StepResultDTO:
        """
        Выполняет один шаг синхронно.

        Args:
            pipeline: Агрегат пайплайна.
            step_num: Номер шага.
            config: Конфигурация пайплайна.

        Returns:
            Результат выполнения шага.
        """
        step_instance = self._registry.get(step_num)
        step_config = StepConfigDTO(
            sequence_number=step_num,
            params={
                "source_path": str(config.source_path),
                "output_path": str(config.output_path),
            },
        )
        try:
            pipeline.start_step(step_num)
            self._save_pipeline(pipeline)
        except Exception as start_err:
            logger.error(f"Cannot start step {step_num}: {start_err}")
            return StepResultDTO(
                sequence_number=step_num,
                status="FAILED",
                message=f"Failed to start step: {start_err}",
                errors=[str(start_err)],
            )

        logger.info(f"Starting step {step_num}: {step_instance.__class__.__name__}")
        try:
            result = step_instance.execute(step_config)
            if result.status == "FAILED":
                pipeline.fail_step(
                    step_num,
                    result.message or "Step returned FAILED status",
                    critical=config.stop_on_error,
                )
                self._save_pipeline(pipeline)
                return result
            else:
                pipeline.complete_step(step_num)
                self._save_pipeline(pipeline)
                return result
        except Exception as e:
            logger.exception(f"Unhandled exception in step {step_num}")
            try:
                pipeline.fail_step(step_num, str(e), critical=config.stop_on_error)
                self._save_pipeline(pipeline)
            except Exception as fail_err:
                logger.error(f"Could not mark step {step_num} as failed: {fail_err}")
                self._save_pipeline(pipeline)
            return StepResultDTO(
                sequence_number=step_num, status="FAILED", message=str(e), errors=[str(e)]
            )

    def _execute_parallel_group(
        self,
        pipeline: PipelineAggregate,
        group_steps: List[int],
        config: PipelineConfigDTO,
    ) -> Generator[StepResultDTO, None, None]:
        """
        Выполняет группу шагов параллельно с учётом уже выполненных.

        При stop_on_error = True после первой ошибки все оставшиеся
        запущенные задачи отменяются.

        Args:
            pipeline: Агрегат пайплайна.
            group_steps: Номера шагов группы в порядке выполнения.
            config: Конфигурация пайплайна.

        Yields:
            Результаты каждого шага по мере завершения.
        """
        pending_steps: List[int] = []
        for step_num in group_steps:
            existing_step = next(
                (s for s in pipeline.steps if s.sequence_number == step_num), None
            )
            if existing_step and existing_step.status == StepStatus.COMPLETED:
                logger.info(f"Step {step_num} already completed, skipping.")
                yield StepResultDTO(
                    sequence_number=step_num,
                    status="SKIPPED",
                    message="Step already completed.",
                )
                continue
            try:
                pipeline.start_step(step_num)
                self._save_pipeline(pipeline)
                pending_steps.append(step_num)
            except Exception as start_err:
                logger.error(f"Cannot start step {step_num}: {start_err}")
                yield StepResultDTO(
                    sequence_number=step_num,
                    status="FAILED",
                    message=f"Failed to start step: {start_err}",
                    errors=[str(start_err)],
                )
                if config.stop_on_error:
                    return

        if not pending_steps:
            return

        with ThreadPoolExecutor(
            max_workers=len(pending_steps), thread_name_prefix="parallel_step"
        ) as executor:
            future_to_step = {}
            for step_num in pending_steps:
                step_instance = self._registry.get(step_num)
                step_config = StepConfigDTO(
                    sequence_number=step_num,
                    params={
                        "source_path": str(config.source_path),
                        "output_path": str(config.output_path),
                    },
                )
                future = executor.submit(step_instance.execute, step_config)
                future_to_step[future] = step_num

            try:
                for future in as_completed(future_to_step):
                    step_num = future_to_step[future]
                    try:
                        result = future.result()
                        if result.status == "FAILED":
                            pipeline.fail_step(
                                step_num,
                                result.message or "Step returned FAILED status",
                                critical=config.stop_on_error,
                            )
                            self._save_pipeline(pipeline)
                            yield result
                            if config.stop_on_error:
                                for f in future_to_step:
                                    f.cancel()
                                return
                        else:
                            pipeline.complete_step(step_num)
                            self._save_pipeline(pipeline)
                            yield result
                    except Exception as e:
                        logger.exception(f"Unhandled exception in step {step_num}")
                        try:
                            pipeline.fail_step(
                                step_num, str(e), critical=config.stop_on_error
                            )
                            self._save_pipeline(pipeline)
                        except Exception as fail_err:
                            logger.error(
                                f"Could not mark step {step_num} as failed: {fail_err}"
                            )
                            self._save_pipeline(pipeline)
                        yield StepResultDTO(
                            sequence_number=step_num,
                            status="FAILED",
                            message=str(e),
                            errors=[str(e)],
                        )
                        if config.stop_on_error:
                            for f in future_to_step:
                                f.cancel()
                            return
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

    def execute(
        self, config: PipelineConfigDTO, pipeline_id: Optional[UUID] = None
    ) -> Generator[StepResultDTO, None, None]:
        """
        Запускает или возобновляет выполнение пайплайна.

        Args:
            config: Конфигурация запуска.
            pipeline_id: ID пайплайна для возобновления. Если None – создаётся новый.

        Yields:
            Результаты шагов по мере завершения.

        Raises:
            ValueError: Если пайплайн с указанным ID не найден.
        """
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
            step_numbers = self._registry.get_step_numbers()

        existing_step_numbers = {s.sequence_number for s in pipeline.steps}
        for step_num in step_numbers:
            if step_num not in existing_step_numbers:
                if step_num in self._registry.get_step_numbers():
                    step_instance = self._registry.get(step_num)
                    step_entity = PipelineStep(
                        sequence_number=step_num,
                        name=step_instance.__class__.__name__,
                    )
                    pipeline.add_step(step_entity)
                else:
                    logger.warning(
                        f"Step {step_num} requested but not found in registry."
                    )
        self._save_pipeline(pipeline)

        completed_parallel: Set[int] = set()
        for step_num in step_numbers:
            if step_num not in {s.sequence_number for s in pipeline.steps}:
                continue

            if step_num in completed_parallel:
                continue

            group = next((g for g in self.PARALLEL_GROUPS if step_num in g), None)
            if group:
                group_steps = [
                    s
                    for s in step_numbers
                    if s in group and s not in completed_parallel
                ]
                yield from self._execute_parallel_group(pipeline, group_steps, config)
                completed_parallel.update(group_steps)
                if pipeline.status == PipelineStatus.FAILED and config.stop_on_error:
                    return
            else:
                existing_step = next(
                    (s for s in pipeline.steps if s.sequence_number == step_num), None
                )
                if existing_step and existing_step.status == StepStatus.COMPLETED:
                    logger.info(f"Step {step_num} already completed, skipping.")
                    yield StepResultDTO(
                        sequence_number=step_num,
                        status="SKIPPED",
                        message="Step already completed.",
                    )
                    continue
                result = self._run_single_step(pipeline, step_num, config)
                yield result
                if result.status == "FAILED" and config.stop_on_error:
                    return