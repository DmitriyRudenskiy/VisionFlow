# src/application/pipeline/steps/color_palette_extraction_step.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.application.pipeline.steps.batch_file_step import BatchFileProcessingStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import StoragePort, ColorPaletteExtractorPort
from src.domain.metadata.value_objects import ColorEntry


class ColorPaletteExtractionStep(BatchFileProcessingStep):
    def __init__(self, storage: StoragePort, extractor: Optional[ColorPaletteExtractorPort] = None):
        super().__init__(storage)
        self._extractor_source = extractor
        self._extractor: Optional[ColorPaletteExtractorPort] = None
        self._num_colors = 20

    def prepare(self) -> None:
        if self._extractor is None:
            if self._extractor_source is not None:
                self._extractor = self._extractor_source
            else:
                from src.infrastructure.ai.color_client import ColorExtractorClient
                self._extractor = ColorExtractorClient()

    @property
    def output_subdirectory(self) -> str:
        return "_colors"

    @property
    def output_suffix(self) -> str:
        return ".json"

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        self._num_colors = config.params.get("num_colors", 20)
        return super().execute(config)

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        if self._extractor is None:
            raise RuntimeError("prepare() was not called before execute()")

        palette_raw = self._extractor.extract_palette(file_path, num_colors=self._num_colors)
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