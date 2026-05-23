# src/application/pipeline/steps/prepare_images_step.py
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import StoragePort
from src.domain.image.value_objects import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class PrepareImagesStep(BaseStep):
    def __init__(self, storage: StoragePort):
        self._storage = storage

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        backup_path = source_path / "_originals"
        self._storage.create_directory(backup_path)

        all_files = self._storage.scan_directory(source_path, recursive=False)
        processed_count = 0
        skipped_count = 0
        rename_counter = 0

        for file_path in all_files:
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped_count += 1
                continue

            backup_dest = backup_path / file_path.name
            if not self._storage.path_exists(backup_dest):
                try:
                    self._storage.copy_file(file_path, backup_dest)
                except OSError as e:
                    logger.warning(f"Failed to backup {file_path.name}: {e}")
                    skipped_count += 1
                    continue

            try:
                mtime = self._storage.get_file_modified_time(file_path)
                timestamp_str = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
                new_stem = f"{timestamp_str}_{rename_counter}"
                new_name = f"{new_stem}{file_path.suffix}"
                dest_path = source_path / new_name

                counter = 1
                while self._storage.path_exists(dest_path):
                    new_name = f"{new_stem}_{counter}{file_path.suffix}"
                    dest_path = source_path / new_name
                    counter += 1

                self._storage.move_file(file_path, dest_path)
                processed_count += 1
                rename_counter += 1
            except OSError as e:
                logger.warning(f"Failed to rename {file_path.name}: {e}")
                skipped_count += 1

        return StepResultDTO.completed(
            sequence_number=config.sequence_number,
            message=f"Prepared {processed_count} images. Backups created. Skipped {skipped_count}.",
            processed_count=processed_count,
            skipped_count=skipped_count,
        )