# application/pipeline/steps/step_7_colors.py
import json
import logging
from pathlib import Path
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort, ColorExtractorPort
from src.domain.image.value_objects import FilePath
from src.domain.metadata.entities import ColorPalette
from src.domain.metadata.value_objects import ColorEntry

logger = logging.getLogger(__name__)


class Step7Colors(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort, color_extractor: ColorExtractorPort):
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
                continue

            try:
                raw_palette = self._color_extractor.extract_palette(file_path, num_colors=5)
                palette = ColorPalette(file_path=FilePath(path=file_path))
                for c in raw_palette:
                    entry = ColorEntry(rgb=tuple(c['rgb']), hex=c['hex'], percentage=c['percentage'])
                    palette.add_color(entry)

                with open(out_json, "w") as f:
                    json.dump({
                        "file": file_path.name,
                        "total_colors": palette.total_colors,
                        "colors": [{"hex": c.hex, "percentage": c.percentage} for c in palette.colors]
                    }, f)

                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to extract colors {file_path.name}: {e}")

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Extracted color palettes for {processed_count} images."
        )