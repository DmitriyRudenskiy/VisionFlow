# src/application/pipeline/steps/batch_file_step.py
from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import StoragePort

logger = logging.getLogger(__name__)


class BatchFileProcessingStep(BaseStep):
    """Базовый класс для шагов, обрабатывающих файлы независимо и сохраняющих результат в JSON."""

    def __init__(self, storage: StoragePort):
        self._storage = storage

    @property
    @abstractmethod
    def output_subdirectory(self) -> str:
        """Имя поддиректории для результатов (например, '_vectors')."""

    @property
    @abstractmethod
    def output_suffix(self) -> str:
        """Суффикс выходного файла (например, '_vector.json')."""

    @abstractmethod
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Обработать файл и вернуть данные для сериализации."""

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        output_dir = source_path / self.output_subdirectory
        self._storage.create_directory(output_dir)

        all_files = self._storage.scan_directory(source_path, recursive=False)

        processed_count = 0
        skipped_count = 0
        error_count = 0
        error_details: List[str] = []

        for file_path in all_files:
            if not file_path.is_file():
                continue

            out_json = output_dir / f"{file_path.stem}{self.output_suffix}"

            # Идемпотентность: пропуск, если файл уже существует
            if self._storage.path_exists(out_json):
                skipped_count += 1
                continue

            try:
                data = self.process_file(file_path)
                if data is None:
                    raise ValueError("Processing returned no data")

                self._storage.persist_json(out_json, data)
                processed_count += 1
            except Exception as e:
                error_msg = f"{file_path.name}: {type(e).__name__}: {e}"
                logger.warning(f"Failed to process {file_path.name} in {self.__class__.__name__}: {e}")
                error_details.append(error_msg)
                error_count += 1

        total = processed_count + skipped_count + error_count

        # Формируем детальное сообщение
        msg_parts = [f"Processed {processed_count}/{total} files"]
        if skipped_count:
            msg_parts.append(f"skipped {skipped_count} (exist)")
        if error_count:
            msg_parts.append(f"failed {error_count}")
            # Добавляем детали ошибок в message
            details = "; ".join(error_details[:3])
            if len(error_details) > 3:
                details += f" (+{len(error_details) - 3} more)"
            msg_parts.append(f"[{details}]")

        return StepResultDTO.completed(
            sequence_number=config.sequence_number,
            message=". ".join(msg_parts) + ".",
            processed_count=processed_count,
            skipped_count=skipped_count,
            errors=error_details if error_details else None,
        )