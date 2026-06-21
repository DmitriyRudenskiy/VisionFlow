# src/application/pipeline/steps/smart_crop_step.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import StoragePort, ImageSegmentationPort

logger = logging.getLogger(__name__)


class SmartCropStep(BaseStep):
    """
    Шаг извлечения объектов.
    Находит все объекты на изображении и сохраняет их как отдельные файлы
    с постфиксом _person_N. Оригинальные файлы не изменяются.
    """

    def __init__(
            self,
            storage: StoragePort,
            segmenter: Optional[ImageSegmentationPort] = None,
            model_path: Optional[str] = None
    ) -> None:
        self._storage = storage
        self._segmenter_factory = segmenter
        self._segmenter: Optional[ImageSegmentationPort] = None
        self._model_path = model_path

    def prepare(self) -> None:
        if self._segmenter is None:
            if self._segmenter_factory is not None:
                self._segmenter = self._segmenter_factory
            else:
                from src.infrastructure.ai.sam3_client import SAM3Client

                model_path = self._model_path
                if not model_path:
                    import sys
                    script_dir = Path(sys.argv[0]).parent.resolve() if hasattr(sys, 'argv') else Path.cwd()
                    models_dir = script_dir.parent / "models"
                    candidates = sorted(models_dir.glob("sam*.pt"))
                    if candidates:
                        model_path = str(candidates[0])
                    else:
                        model_path = str(models_dir / "sam3.pt")

                self._segmenter = SAM3Client(model_path=model_path)

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        if self._segmenter is None:
            raise RuntimeError("prepare() was not called before execute()")

        source_path = Path(config.params.get("source_path", "."))
        output_path = Path(config.params.get("output_path", "."))  # Output path from config

        if not source_path.is_dir():
            return StepResultDTO.failed(
                sequence_number=config.sequence_number,
                message=f"Source path is not a directory: {source_path}",
            )

        # Убедимся, что выходная директория существует
        self._storage.create_directory(output_path)

        mode = config.params.get("crop_mode", "square")
        if mode not in ["square", "mask", "transparent"]:
            return StepResultDTO.failed(
                sequence_number=config.sequence_number,
                message=f"Unsupported crop mode: {mode}",
            )

        all_files = self._storage.scan_directory(source_path, recursive=False)
        total_cropped = 0
        errors: list[str] = []

        for file_path in all_files:
            if not file_path.is_file():
                continue

            try:
                # Получаем список временных файлов с кропами
                cropped_paths = self._segmenter.crop_image(file_path, mode=mode)

                if not cropped_paths:
                    continue

                # Сохраняем каждый кроп с уникальным именем
                for idx, crop_tmp_path in enumerate(cropped_paths):
                    # Формируем имя: item_00001_person_0.jpg
                    # Если объект один, можно без индекса, но для единообразия оставим индекс или проверку
                    suffix_str = f"_person_{idx}" if len(cropped_paths) > 1 else "_person"

                    # Определяем расширение по результату (jpg или png)
                    new_name = f"{file_path.stem}{suffix_str}{crop_tmp_path.suffix}"
                    dest_path = output_path / new_name

                    try:
                        self._storage.move_file(crop_tmp_path, dest_path)
                        logger.info(f"Saved: {dest_path.name}")
                        total_cropped += 1
                    except Exception as move_err:
                        logger.warning(f"Failed to move crop {idx} for {file_path.name}: {move_err}")
                        errors.append(f"{file_path.name} crop {idx}: move failed")
                        # Удаляем временный файл, если не смогли переместить
                        if crop_tmp_path.exists():
                            crop_tmp_path.unlink()

            except Exception as e:
                err_msg = f"Failed to process {file_path.name}: {e}"
                logger.warning(err_msg)
                errors.append(err_msg)

        msg = f"Extracted {total_cropped} objects."
        if errors:
            msg += f" Errors: {len(errors)}."

        return StepResultDTO.completed(
            sequence_number=config.sequence_number,
            message=msg,
            processed_count=total_cropped,
            errors=errors if errors else None,
        )