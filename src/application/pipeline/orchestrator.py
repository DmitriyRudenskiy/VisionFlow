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
    Управляет жизненным циклом, параллелизмом и сохранением состояния.
    """

    # Группы шагов, которые можно выполнять параллельно.
    PARALLEL_GROUPS: List[Set[int]] = [{5, 6, 7, 8}]

    def __init__(self, step_registry: StepRegistry, pipeline_repo: PipelineRepository) -> None:
        self._registry = step_registry
        self._repo = pipeline_repo
        self._save_lock = threading.RLock()

    def get_available_steps(self) -> List[int]:
        return self._registry.get_step_numbers()

    def create_pipeline(self, source_path: Path, output_path: Path) -> PipelineAggregate:
        return PipelineAggregate(
            name=f"Pipeline-{source_path.name}",
            source_directory=source_path,
            output_directory=output_path,
        )

    def _save_pipeline(self, pipeline: PipelineAggregate) -> None:
        with self._save_lock:
            self._repo.save(pipeline)

    def _build_step_config(self, step_num: int, config: PipelineConfigDTO) -> StepConfigDTO:
        return StepConfigDTO(
            sequence_number=step_num,
            params={
                "source_path": str(config.source_path),
                "output_path": str(config.output_path),
            },
        )

    def _create_execution_plan(self, requested_steps: List[int]) -> List[List[int]]:
        """
        Создает план выполнения: список батчей.
        Батч с одним шагом выполняется последовательно.
        Батч с несколькими шагами выполняется параллельно.
        """
        plan: List[List[int]] = []
        processed_steps: Set[int] = set()

        for step_num in requested_steps:
            if step_num in processed_steps:
                continue

            # Проверяем, принадлежит ли шаг параллельной группе
            parallel_group = next((g for g in self.PARALLEL_GROUPS if step_num in g), None)

            if parallel_group:
                # Собираем все шаги из этой группы, которые были запрошены
                batch = sorted(list(set(requested_steps) & parallel_group))
                plan.append(batch)
                processed_steps.update(batch)
            else:
                plan.append([step_num])
                processed_steps.add(step_num)

        return plan

    def _execute_single_step_logic(self, step_num: int, config: PipelineConfigDTO) -> StepResultDTO:
        """Непосредственное выполнение логики шага."""
        step_instance = self._registry.get(step_num)
        step_instance.prepare()
        step_config = self._build_step_config(step_num, config)
        return step_instance.execute(step_config)

    def _handle_step_result(
            self, pipeline: PipelineAggregate, step_num: int, result: StepResultDTO, stop_on_error: bool
    ) -> None:
        """Обновление состояния агрегата на основе результата."""
        with self._save_lock:
            if result.status == "FAILED":
                pipeline.fail_step(step_num, result.message or "Unknown error", critical=stop_on_error)
            else:
                pipeline.complete_step(step_num)
            self._repo.save(pipeline)

    def _run_batch(
            self, pipeline: PipelineAggregate, batch: List[int], config: PipelineConfigDTO
    ) -> Generator[StepResultDTO, None, None]:
        """Выполняет батч шагов (последовательно или параллельно)."""

        # Фильтруем уже завершенные шаги
        pending_steps: List[int] = []
        for step_num in batch:
            step = pipeline.find_step(step_num)
            if step and step.status == StepStatus.COMPLETED:
                yield StepResultDTO.skipped(step_num)
            else:
                pending_steps.append(step_num)

        if not pending_steps:
            return

        # Если шаг один - выполняем синхронно
        if len(pending_steps) == 1:
            step_num = pending_steps[0]
            try:
                pipeline.start_step(step_num)
                self._save_pipeline(pipeline)

                result = self._execute_single_step_logic(step_num, config)
                self._handle_step_result(pipeline, step_num, result, config.stop_on_error)
                yield result
            except Exception as e:
                logger.exception(f"Critical error in step {step_num}")
                result = StepResultDTO.failed(step_num, str(e), [str(e)])
                self._handle_step_result(pipeline, step_num, result, config.stop_on_error)
                yield result
            return

        # Параллельное выполнение
        # Сначала помечаем все как RUNNING
        for step_num in pending_steps:
            try:
                pipeline.start_step(step_num)
            except Exception as e:
                yield StepResultDTO.failed(step_num, f"Failed to start: {e}", [str(e)])
                return  # Если не можем стартануть, останавливаем батч

        self._save_pipeline(pipeline)

        executor = ThreadPoolExecutor(max_workers=len(pending_steps), thread_name_prefix="parallel_step")
        try:
            future_map = {
                executor.submit(self._execute_single_step_logic, num, config): num
                for num in pending_steps
            }

            for future in as_completed(future_map):
                step_num = future_map[future]
                try:
                    result = future.result()
                    self._handle_step_result(pipeline, step_num, result, config.stop_on_error)
                    yield result
                except Exception as e:
                    logger.exception(f"Unhandled exception in parallel step {step_num}")
                    result = StepResultDTO.failed(step_num, str(e), [str(e)])
                    self._handle_step_result(pipeline, step_num, result, config.stop_on_error)
                    yield result
                    if config.stop_on_error:
                        # Не ждем оставшиеся задачи — завершаем executor немедленно
                        executor.shutdown(wait=False)
                        return
        finally:
            # Гарантируем закрытие пула, если он ещё не завершён
            executor.shutdown(wait=False)

    def execute(
            self, config: PipelineConfigDTO, pipeline_id: Optional[str | UUID] = None
    ) -> Generator[StepResultDTO, None, None]:
        """Запускает или возобновляет выполнение пайплайна."""

        # 1. Инициализация или загрузка
        if not config.source_path.exists():
            raise ValueError(f"Source path does not exist: {config.source_path}")

        if pipeline_id:
            _pid = pipeline_id if isinstance(pipeline_id, UUID) else UUID(pipeline_id)
            pipeline = self._repo.find_by_id(_pid)
            if not pipeline:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            pipeline.resume()
        else:
            pipeline = self.create_pipeline(config.source_path, config.output_path)

        self._save_pipeline(pipeline)

        # 2. Определение списка шагов
        if config.steps_to_run is not None:
            target_steps = sorted(config.steps_to_run)
        elif pipeline_id:
            target_steps = sorted([s.sequence_number for s in pipeline.steps])
        else:
            target_steps = self._registry.get_step_numbers()

        # 3. Регистрация шагов в агрегате (если их нет)
        available_steps = self._registry.get_step_numbers()
        for step_num in target_steps:
            if not pipeline.find_step(step_num) and step_num in available_steps:
                step_name = self._registry.get(step_num).__class__.__name__
                pipeline.add_step(PipelineStep(sequence_number=step_num, name=step_name))

        self._save_pipeline(pipeline)

        # 4. Планирование и выполнение
        execution_plan = self._create_execution_plan(target_steps)

        for batch in execution_plan:
            if pipeline.status == PipelineStatus.FAILED and config.stop_on_error:
                break

            yield from self._run_batch(pipeline, batch, config)