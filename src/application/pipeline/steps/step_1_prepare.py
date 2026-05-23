# application/pipeline/steps/step_1_prepare.py
import logging
from pathlib import Path
from datetime import datetime
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort
from src.domain.image.value_objects import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class Step1Prepare(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort):
        self._fs = fs_service

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        backup_path = source_path / "_originals"
        self._fs.create_directory(backup_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)
        processed_count = 0
        skipped_count = 0

        for file_path in all_files:
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped_count += 1
                continue

            backup_dest = backup_path / file_path.name
            if not backup_dest.exists():
                try:
                    self._fs.copy_file(file_path, backup_dest)
                except OSError as e:
                    logger.warning(f"Failed to backup {file_path.name}: {e}")
                    skipped_count += 1
                    continue

            try:
                mtime = self._fs.get_file_modified_time(file_path)
                timestamp_str = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
                new_stem = f"{timestamp_str}_{processed_count}"
                new_name = f"{new_stem}{file_path.suffix}"
                dest_path = source_path / new_name

                counter = 1
                while dest_path.exists():
                    new_name = f"{new_stem}_{counter}{file_path.suffix}"
                    dest_path = source_path / new_name
                    counter += 1

                self._fs.move_file(file_path, dest_path)
                processed_count += 1
            except OSError as e:
                logger.warning(f"Failed to rename {file_path.name}: {e}")
                skipped_count += 1

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Prepared {processed_count} images. Backups created. Skipped {skipped_count}.",
            processed_count=processed_count,
            skipped_count=skipped_count
        )