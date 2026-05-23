# src/application/pipeline/steps/step_4_ai_crop.py
import logging
from pathlib import Path
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort, AISegmenterPort

logger = logging.getLogger(__name__)


class Step4AICrop(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort, ai_segmenter: AISegmenterPort):
        self._fs = fs_service
        self._segmenter = ai_segmenter

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))

        mode = config.params.get("crop_mode", "square")
        if mode not in ["square", "mask", "transparent"]:
            raise ValueError(f"Unsupported crop mode: {mode}")

        all_files = self._fs.scan_directory(source_path, recursive=False)
        cropped_count = 0

        for file_path in all_files:
            if not file_path.is_file():
                continue

            try:
                cropped_image_path = self._segmenter.crop_image(file_path, mode=mode)

                if cropped_image_path.exists():
                    # If the segmenter returned a different path (temp file), move it to source
                    if cropped_image_path != file_path:
                        # Resolve naming conflict if necessary, or overwrite
                        # We assume we want to keep the cropped version in the main flow
                        dest_path = source_path / cropped_image_path.name

                        # Handling name collision if name differs from original but exists
                        if dest_path != file_path and dest_path.exists():
                            # For simplicity in this step, we assume segmenter generates unique name
                            # or we overwrite. Here we move.
                            pass

                        self._fs.move_file(cropped_image_path, dest_path)

                    cropped_count += 1
            except Exception as e:
                logger.warning(f"Failed to crop {file_path.name}: {e}")

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Successfully cropped {cropped_count} images in '{mode}' mode."
        )