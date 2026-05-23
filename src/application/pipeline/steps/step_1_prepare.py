# src/application/pipeline/steps/step_1_prepare.py
from pathlib import Path
from datetime import datetime
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort


class Step1Prepare(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort):
        self._fs = fs_service

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        backup_path = source_path / "_originals"
        self._fs.create_directory(backup_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)
        processed_count = 0

        for file_path in all_files:
            if file_path.parent == backup_path or file_path.is_dir():
                continue

            # Бэкап оригиналов
            backup_dest = backup_path / file_path.name
            self._fs.copy_file(file_path, backup_dest)

            # Переименование по timestamp, с сохранением оригинального расширения
            mtime = self._fs.get_file_modified_time(file_path)
            timestamp_str = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
            new_stem = f"{timestamp_str}_{processed_count}"
            new_name = f"{new_stem}{file_path.suffix}"  # используем исходный суффикс
            dest_path = source_path / new_name

            # Проверка на существование и добавление суффикса при необходимости
            counter = 1
            while dest_path.exists():
                new_name = f"{new_stem}_{counter}{file_path.suffix}"
                dest_path = source_path / new_name
                counter += 1

            self._fs.move_file(file_path, dest_path)
            processed_count += 1

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Prepared {processed_count} images. Backups created."
        )