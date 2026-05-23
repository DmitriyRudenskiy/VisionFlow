# src/application/pipeline/orchestrator.py
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

    def _build_step_config(self, step_num: int, config: PipelineConfigDTO) -> StepConfigDTO:
        """Фабричный метод для создания конфигурации шага."""
        return StepConfigDTO(
            sequence_number=step_num,
            params={
                "source_path": str(config.source_path),
                "output_path": str(config.output_path),
            },
        )

    def _execute_step_logic(
            self, step_num: int, config: PipelineConfigDTO
    ) -> StepResultDTO:
        """Непосредственное выполнение логики шага (без управления состоянием агрегата)."""
        step_instance = self._registry.get(step_num)
        step_config = self._build_step_config(step_num, config)
        return step_instance.execute(step_config)

    def _process_step_result(
            self,
            pipeline: PipelineAggregate,
            step_num: int,
            result: StepResultDTO,
            stop_on_error: bool
    ) -> None:
        """Обработка результата шага и обновление состояния пайплайна."""
        if result.status == "FAILED":
            pipeline.fail_step(
                step_num,
                result.message or "Step returned FAILED status",
                critical=stop_on_error,
            )
        else:
            pipeline.complete_step(step_num)
        self._save_pipeline(pipeline)

    def _run_single_step(
            self, pipeline: PipelineAggregate, step_num: int, config: PipelineConfigDTO
    ) -> StepResultDTO:
        """Выполняет один шаг синхронно с управлением состояния."""
        try:
            pipeline.start_step(step_num)
            self._save_pipeline(pipeline)
        except Exception as start_err:
            logger.error(f"Cannot start step {step_num}: {start_err}")
            return StepResultDTO.failed(step_num, f"Failed to start step: {start_err}", [str(start_err)])

        logger.info(f"Starting step {step_num}")
        try:
            result = self._execute_step_logic(step_num, config)
            self._process_step_result(pipeline, step_num, result, config.stop_on_error)
            return result
        except Exception as e:
            logger.exception(f"Unhandled exception in step {step_num}")
            try:
                pipeline.fail_step(step_num, str(e), critical=config.stop_on_error)
                self._save_pipeline(pipeline)
            except Exception as fail_err:
                logger.error(f"Could not mark step {step_num} as failed: {fail_err}")
            return StepResultDTO.failed(step_num, str(e), [str(e)])

    def _execute_parallel_group(
            self,
            pipeline: PipelineAggregate,
            group_steps: List[int],
            config: PipelineConfigDTO,
    ) -> Generator[StepResultDTO, None, None]:
        """Выполняет группу шагов параллельно."""

        # 1. Подготовка и фильтрация уже выполненных шагов
        pending_steps: List[int] = []
        for step_num in group_steps:
            existing_step = pipeline.find_step(step_num)
            if existing_step and existing_step.status == StepStatus.COMPLETED:
                yield StepResultDTO.skipped(step_num)
                continue

            try:
                pipeline.start_step(step_num)
                self._save_pipeline(pipeline)
                pending_steps.append(step_num)
            except Exception as start_err:
                yield StepResultDTO.failed(step_num, f"Failed to start: {start_err}", [str(start_err)])
                if config.stop_on_error:
                    return

        if not pending_steps:
            return

        # 2. Параллельное выполнение
        with ThreadPoolExecutor(max_workers=len(pending_steps), thread_name_prefix="parallel_step") as executor:
            future_to_step = {
                executor.submit(self._execute_step_logic, num, config): num
                for num in pending_steps
            }

            try:
                for future in as_completed(future_to_step):
                    step_num = future_to_step[future]
                    try:
                        result = future.result()
                        self._process_step_result(pipeline, step_num, result, config.stop_on_error)
                        yield result

                        if result.status == "FAILED" and config.stop_on_error:
                            # Отмена оставшихся задач при ошибке
                            for f in future_to_step:
                                f.cancel()
                            return
                    except Exception as e:
                        logger.exception(f"Unhandled exception in parallel step {step_num}")
                        pipeline.fail_step(step_num, str(e), critical=config.stop_on_error)
                        self._save_pipeline(pipeline)
                        yield StepResultDTO.failed(step_num, str(e), [str(e)])
                        if config.stop_on_error:
                            for f in future_to_step:
                                f.cancel()
                            return
            finally:
                # Явное завершение работы executor'а
                executor.shutdown(wait=False, cancel_futures=True)

    def execute(
            self, config: PipelineConfigDTO, pipeline_id: Optional[UUID] = None
    ) -> Generator[StepResultDTO, None, None]:
        """Запускает или возобновляет выполнение пайплайна."""
        if pipeline_id:
            pipeline = self._repo.find_by_id(pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline with ID {pipeline_id} not found")
            pipeline.resume()
        else:
            pipeline = self.create_pipeline(config.source_path, config.output_path)
            self._save_pipeline(pipeline)

        # Определение списка шагов для выполнения
        if config.steps_to_run is not None:
            step_numbers = sorted(config.steps_to_run)
        elif pipeline_id:
            step_numbers = sorted([s.sequence_number for s in pipeline.steps])
        else:
            step_numbers = self._registry.get_step_numbers()

        # Добавление шагов в агрегат, если их там нет
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
                    logger.warning(f"Step {step_num} requested but not found in registry.")
        self._save_pipeline(pipeline)

        # Основной цикл выполнения
        completed_parallel: Set[int] = set()
        for step_num in step_numbers:
            # Пропуск уже выполненных в рамках этой сессии параллельных групп
            if step_num in completed_parallel:
                continue

            # Проверка статуса пайплайна (если упали ранее)
            if pipeline.status == PipelineStatus.FAILED and config.stop_on_error:
                return

            # Определение группы параллельности
            group = next((g for g in self.PARALLEL_GROUPS if step_num in g), None)

            if group:
                # Исключаем шаги, которые уже могли быть выполнены или запланированы
                group_steps = [s for s in step_numbers if s in group and s not in completed_parallel]
                if not group_steps:
                    continue

                yield from self._execute_parallel_group(pipeline, group_steps, config)
                completed_parallel.update(group_steps)
            else:
                # Последовательное выполнение
                existing_step = pipeline.find_step(step_num)
                if existing_step and existing_step.status == StepStatus.COMPLETED:
                    yield StepResultDTO.skipped(step_num)
                    continue

                result = self._run_single_step(pipeline, step_num, config)
                yield result
                if result.status == "FAILED" and config.stop_on_error:
                    return