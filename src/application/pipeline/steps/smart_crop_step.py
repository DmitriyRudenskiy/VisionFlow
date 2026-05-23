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
    def __init__(self, storage: StoragePort, segmenter: Optional[ImageSegmentationPort] = None):
        self._storage = storage
        self._segmenter_factory = segmenter
        self._segmenter: Optional[ImageSegmentationPort] = None

    def prepare(self) -> None:
        if self._segmenter is None:
            if self._segmenter_factory is not None:
                self._segmenter = self._segmenter_factory
            else:
                from src.infrastructure.ai.sam3_client import SAM3Client
                import sys
                from pathlib import Path
                script_dir = Path(sys.argv[0]).parent.resolve() if hasattr(sys, 'argv') else Path.cwd()
                sam3_checkpoint = script_dir.parent / "models" / "sam3.pt"
                self._segmenter = SAM3Client(model_path=str(sam3_checkpoint))

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        if self._segmenter is None:
            raise RuntimeError("prepare() was not called before execute()")

        source_path = Path(config.params.get("source_path", "."))
        if not source_path.is_dir():
            return StepResultDTO.failed(
                sequence_number=config.sequence_number,
                message=f"Source path is not a directory: {source_path}",
            )

        mode = config.params.get("crop_mode", "square")
        if mode not in ["square", "mask", "transparent"]:
            return StepResultDTO.failed(
                sequence_number=config.sequence_number,
                message=f"Unsupported crop mode: {mode}",
            )

        all_files = self._storage.scan_directory(source_path, recursive=False)
        cropped_count = 0
        errors: list[str] = []

        for file_path in all_files:
            if not file_path.is_file():
                continue

            try:
                cropped_image_path = self._segmenter.crop_image(file_path, mode=mode)

                if cropped_image_path and self._storage.path_exists(cropped_image_path):
                    if cropped_image_path != file_path:
                        # Заменяем оригинал кропнутой версией
                        try:
                            self._storage.move_file(cropped_image_path, file_path)
                        except Exception as move_err:
                            logger.warning(
                                f"Failed to replace original {file_path.name} with crop: {move_err}"
                            )
                            errors.append(f"{file_path.name}: move failed: {move_err}")
                            continue
                    cropped_count += 1
                else:
                    logger.warning(
                        f"Segmenter returned non-existent path for {file_path.name}: {cropped_image_path}"
                    )
            except Exception as e:
                err_msg = f"Failed to crop {file_path.name}: {e}"
                logger.warning(err_msg)
                errors.append(err_msg)

        msg = f"Successfully cropped {cropped_count} images in '{mode}' mode."
        if errors:
            msg += f" Errors: {len(errors)}."

        return StepResultDTO.completed(
            sequence_number=config.sequence_number,
            message=msg,
            processed_count=cropped_count,
            errors=errors if errors else None,
        )