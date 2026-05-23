import logging
from pathlib import Path
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import StoragePort, ImageSegmentationPort

logger = logging.getLogger(__name__)


class SmartCropStep(BaseStep):
    def __init__(self, storage: StoragePort, segmenter: ImageSegmentationPort):
        self._storage = storage
        self._segmenter = segmenter

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))

        mode = config.params.get("crop_mode", "square")
        if mode not in ["square", "mask", "transparent"]:
            raise ValueError(f"Unsupported crop mode: {mode}")

        all_files = self._storage.scan_directory(source_path, recursive=False)
        cropped_count = 0

        for file_path in all_files:
            if not file_path.is_file():
                continue

            try:
                cropped_image_path = self._segmenter.crop_image(file_path, mode=mode)

                if cropped_image_path and self._storage.path_exists(cropped_image_path):
                    if cropped_image_path != file_path:
                        dest_path = source_path / cropped_image_path.name

                        if self._storage.path_exists(dest_path) and dest_path != file_path:
                            stem = dest_path.stem
                            suffix = dest_path.suffix
                            counter = 1
                            new_dest = dest_path
                            while self._storage.path_exists(new_dest):
                                new_dest = source_path / f"{stem}_{counter}{suffix}"
                                counter += 1
                            dest_path = new_dest

                        self._storage.move_file(cropped_image_path, dest_path)

                    cropped_count += 1
                else:
                    logger.warning(
                        f"Segmenter returned non-existent path for {file_path.name}"
                    )
            except Exception as e:
                logger.warning(f"Failed to crop {file_path.name}: {e}")

        return StepResultDTO.completed(
            sequence_number=config.sequence_number,
            message=f"Successfully cropped {cropped_count} images in '{mode}' mode.",
            processed_count=cropped_count,
        )