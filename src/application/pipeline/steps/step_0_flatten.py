# application/pipeline/steps/step_0_flatten.py
import logging
from pathlib import Path
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort

logger = logging.getLogger(__name__)


class Step0Flatten(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort):
        self._fs = fs_service

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))

        all_files = self._fs.scan_directory(source_path, recursive=True)
        moved_count = 0
        skipped_count = 0

        for file_path in all_files:
            # Файл уже в корне — пропускаем
            if file_path.parent == source_path:
                continue

            dest_path = source_path / file_path.name
            if dest_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = source_path / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                self._fs.move_file(file_path, dest_path)
                moved_count += 1
            except OSError as e:
                logger.warning(f"Failed to move {file_path} to {dest_path}: {e}")
                skipped_count += 1

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Flattened directory. Moved {moved_count} files to root. Skipped {skipped_count}.",
            processed_count=moved_count,
            skipped_count=skipped_count,
        )
