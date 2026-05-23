# src/application/pipeline/steps/batch_file_step.py
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict

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

        for file_path in all_files:
            if not file_path.is_file():
                continue

            out_json = output_dir / f"{file_path.stem}{self.output_suffix}"

            # Идемпотентность: пропуск, если файл уже существует
            if self._storage.path_exists(out_json):
                processed_count += 1
                continue

            try:
                data = self.process_file(file_path)
                if data is None:
                    raise ValueError("Processing returned no data")

                self._storage.persist_json(out_json, data)
                processed_count += 1
            except Exception as e:
                logger.warning(
                    f"Failed to process {file_path.name} in {self.__class__.__name__}: {e}"
                )
                skipped_count += 1

        return StepResultDTO.completed(
            sequence_number=config.sequence_number,
            message=f"Processed {processed_count} files. Skipped {skipped_count}.",
            processed_count=processed_count,
            skipped_count=skipped_count,
        )