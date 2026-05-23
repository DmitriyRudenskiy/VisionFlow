from pathlib import Path
from typing import Any, Dict

from src.application.pipeline.steps.batch_file_step import BatchFileProcessingStep
from src.application.ports import StoragePort, ImageEmbeddingExtractorPort
from src.domain.metadata.value_objects import VectorEmbedding


class EmbeddingExtractionStep(BatchFileProcessingStep):
    def __init__(self, storage: StoragePort, extractor: ImageEmbeddingExtractorPort):
        super().__init__(storage)
        self._extractor = extractor

    @property
    def output_subdirectory(self) -> str:
        return "_vectors"

    @property
    def output_suffix(self) -> str:
        return "_vector.json"

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        raw_vector = self._extractor.get_embedding(file_path)
        if not raw_vector:
            raise ValueError("Empty embedding received")
        embedding = VectorEmbedding(values=tuple(raw_vector), model_name="Qwen-VL")
        normalized = embedding.l2_normalize()
        return {
            "file": file_path.name,
            "model": normalized.model_name,
            "vector": list(normalized.values),
            "dimension": normalized.dimension,
        }