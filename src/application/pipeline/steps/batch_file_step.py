# src/application/pipeline/steps/batch_file_step.py
from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import StoragePort

logger = logging.getLogger(__name__)


class BatchFileProcessingStep(BaseStep):
    """
    Базовый класс для шагов, обрабатывающих файлы независимо.
    Автоматически создает выходную директорию, пропускает существующие файлы
    и собирает ошибки.
    """

    def __init__(self, storage: StoragePort) -> None:
        self._storage = storage

    @property
    @abstractmethod
    def output_subdirectory(self) -> str:
        """Например: '_vectors'"""

    @property
    @abstractmethod
    def output_suffix(self) -> str:
        """Например: '_vector.json'"""

    @abstractmethod
    def process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Обработать один файл.
        Возвращает словарь для сохранения в JSON или None, если результат сохранить нельзя.
        """

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        if not source_path.is_dir():
            return StepResultDTO.failed(config.sequence_number, f"Not a directory: {source_path}")

        output_dir = source_path / self.output_subdirectory
        if output_dir.exists() and output_dir.is_file():
            return StepResultDTO.failed(
                config.sequence_number,
                f"Cannot create output directory: {output_dir} is a file"
            )
        self._storage.create_directory(output_dir)

        files = self._storage.scan_directory(source_path, recursive=False)

        processed, skipped, errors = 0, 0, 0
        error_details: List[str] = []

        force_overwrite = config.params.get("force_overwrite", False)

        for file_path in files:
            out_path = output_dir / f"{file_path.stem}{self.output_suffix}"

            if self._storage.path_exists(out_path) and not force_overwrite:
                skipped += 1
                continue

            try:
                data = self.process_file(file_path)
                if data:
                    self._storage.persist_json(out_path, data)
                    processed += 1
                else:
                    logger.warning(f"No data returned for {file_path.name}")

            except Exception as e:
                errors += 1
                msg = f"{file_path.name}: {e}"
                logger.warning(f"Processing error: {msg}")
                error_details.append(msg)
                # Обрываем выполнение, если накопилось много ошибок (опционально)
                if errors > 10:
                    logger.error("Too many errors, aborting step.")
                    error_details.append("Aborted due to multiple errors.")
                    break

        msg = f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}."
        return StepResultDTO.completed(
            config.sequence_number, message=msg,
            processed_count=processed, skipped_count=skipped,
            errors=error_details or None
        )