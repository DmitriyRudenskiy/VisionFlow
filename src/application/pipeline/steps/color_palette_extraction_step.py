from pathlib import Path
from typing import Any, Dict

from src.application.pipeline.steps.batch_file_step import BatchFileProcessingStep
from src.application.ports import StoragePort, ColorPaletteExtractorPort
from src.domain.metadata.value_objects import ColorEntry


class ColorPaletteExtractionStep(BatchFileProcessingStep):
    def __init__(self, storage: StoragePort, extractor: ColorPaletteExtractorPort):
        super().__init__(storage)
        self._extractor = extractor

    @property
    def output_subdirectory(self) -> str:
        return "_colors"

    @property
    def output_suffix(self) -> str:
        return "_colors.json"

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        palette_raw = self._extractor.extract_palette(file_path)
        color_entries = [
            ColorEntry(rgb=tuple(c["rgb"]), hex=c["hex"], percentage=c["percentage"])
            for c in palette_raw
        ]
        return {
            "file": file_path.name,
            "colors": [
                {"rgb": list(c.rgb), "hex": c.hex, "percentage": c.percentage}
                for c in color_entries
            ],
        }