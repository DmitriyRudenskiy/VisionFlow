# application/pipeline/steps/step_3_visual_dups.py
from collections import defaultdict
from pathlib import Path
import logging
import html

from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort, VisualDupDetectorPort
from src.domain.image.value_objects import FilePath
from src.domain.image.exceptions import InvalidImageFormat
from src.domain.deduplication.value_objects import FileHash
from src.domain.deduplication.entities import HashEntry, DuplicateGroup

logger = logging.getLogger(__name__)


class Step3VisualDups(BaseStep):
    def __init__(
        self, fs_service: FileSystemServicePort, visual_detector: VisualDupDetectorPort
    ):
        self._fs = fs_service
        self._detector = visual_detector

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        visual_dups_path = source_path / "_visual_duplicates"
        self._fs.create_directory(visual_dups_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)

        hash_map = defaultdict(list)
        skipped_count = 0
        for file_path in all_files:
            try:
                file_vo = FilePath(path=file_path)
            except (InvalidImageFormat, ValueError) as e:
                logger.warning(f"Skipping file {file_path.name}: {e}")
                skipped_count += 1
                continue
            try:
                phash = self._detector.calculate_phash(file_path)
            except Exception:
                logger.warning(
                    f"Skipping file {file_path.name}, failed to calculate phash"
                )
                skipped_count += 1
                continue
            modified_at = self._fs.get_file_modified_time(file_path)
            entry = HashEntry(
                file_hash=FileHash(algorithm="phash", value=phash),
                file_path=file_vo,
                file_size=self._fs.get_file_size(file_path),
                modified_at=modified_at,
            )
            hash_map[phash].append(entry)

        groups_found = 0
        html_items = []

        for hash_value, entries in hash_map.items():
            if len(entries) < 2:
                continue

            group = DuplicateGroup(group_id=f"phash_{hash_value}", entries=entries)
            group.resolve_original_by_size()

            if group.original_entry:
                groups_found += 1
                orig_name = html.escape(group.original_entry.file_path.name)
                html_items.append(
                    f"<div><h3>Group {html.escape(group.group_id)}</h3><p>Original: {orig_name}</p></div>"
                )

                for dup in group.duplicates:
                    dest_path = visual_dups_path / dup.file_path.name
                    self._fs.move_file(dup.file_path.path, dest_path)

        report_path = source_path / "visual_dups_report.html"
        html_content = (
            "<html><head><meta charset='utf-8'><title>Visual Duplicates Report</title></head>"
            "<body><h1>Visual Duplicates Report</h1>"
            f"<p>Total groups found: {groups_found}</p>"
            f"{''.join(html_items)}</body></html>"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Found {groups_found} visual duplicate groups. Report saved. Skipped {skipped_count} files.",
        )
