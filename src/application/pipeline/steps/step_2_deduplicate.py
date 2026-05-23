from collections import defaultdict
from pathlib import Path
import logging

from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort
from src.domain.image.value_objects import FilePath
from src.domain.deduplication.value_objects import FileHash
from src.domain.deduplication.entities import HashEntry, DuplicateGroup
from src.domain.image.exceptions import InvalidImageFormat

logger = logging.getLogger(__name__)


class Step2Deduplicate(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort):
        self._fs = fs_service

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        duplicates_path = source_path / "_duplicates"
        self._fs.create_directory(duplicates_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)

        hash_map = defaultdict(list)
        skipped_count = 0

        for file_path in all_files:
            if file_path.is_dir() or file_path.parent == duplicates_path:
                continue

            try:
                file_vo = FilePath(path=file_path)

                hash_value = self._fs.get_file_hash(file_path, algorithm="md5")
                file_size = self._fs.get_file_size(file_path)
                modified_at = self._fs.get_file_modified_time(file_path)  # получаем время модификации

                entry = HashEntry(
                    file_hash=FileHash(algorithm="md5", value=hash_value),
                    file_path=file_vo,
                    file_size=file_size,
                    modified_at=modified_at
                )
                hash_map[hash_value].append(entry)

            except InvalidImageFormat:
                logger.info(f"Skipping non-image file: {file_path.name}")
                skipped_count += 1
            except Exception as e:
                logger.error(f"Error processing file {file_path.name}: {e}")
                skipped_count += 1

        duplicates_found = 0

        for hash_value, entries in hash_map.items():
            if len(entries) > 1:
                group = DuplicateGroup(group_id=hash_value[:8], entries=entries)
                group.resolve_original_by_size()

                for dup in group.duplicates:
                    dest_path = duplicates_path / dup.file_path.name
                    self._fs.move_file(dup.file_path.path, dest_path)
                    duplicates_found += 1

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Found {duplicates_found} duplicates. Skipped {skipped_count} files."
        )