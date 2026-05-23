# src/application/pipeline/steps/step_7_colors.py
import json
import logging
from pathlib import Path

from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort, ColorExtractorPort
from src.domain.metadata.value_objects import ColorEntry

logger = logging.getLogger(__name__)


class Step7Colors(BaseStep):
    def __init__(
        self, fs_service: FileSystemServicePort, color_extractor: ColorExtractorPort
    ):
        self._fs = fs_service
        self._color_extractor = color_extractor

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        colors_path = source_path / "_colors"
        self._fs.create_directory(colors_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)
        processed_count = 0

        for file_path in all_files:
            if not file_path.is_file():
                continue

            out_json = colors_path / f"{file_path.stem}_colors.json"
            if out_json.exists():
                processed_count += 1
                continue

            try:
                palette_raw = self._color_extractor.extract_palette(file_path)
                # Преобразование в доменные объекты (опционально, здесь для целостности)
                color_entries = [
                    ColorEntry(
                        rgb=tuple(c["rgb"]), hex=c["hex"], percentage=c["percentage"]
                    )
                    for c in palette_raw
                ]

                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "file": file_path.name,
                            "colors": [
                                {
                                    "rgb": list(c.rgb),
                                    "hex": c.hex,
                                    "percentage": c.percentage,
                                }
                                for c in color_entries
                            ],
                        },
                        f,
                        ensure_ascii=False,
                    )

                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to extract colors for {file_path.name}: {e}")

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Extracted color palettes for {processed_count} images.",
            processed_count=processed_count,
        )
