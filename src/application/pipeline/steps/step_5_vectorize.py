# application/pipeline/steps/step_5_vectorize.py
import json
import logging
from pathlib import Path
from typing import List
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort, VectorizationPort
from src.domain.metadata.value_objects import VectorEmbedding

logger = logging.getLogger(__name__)


class Step5Vectorize(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort, vectorizer: VectorizationPort):
        self._fs = fs_service
        self._vectorizer = vectorizer

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        vectors_path = source_path / "_vectors"
        self._fs.create_directory(vectors_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)
        processed_count = 0

        for file_path in all_files:
            if not file_path.is_file():
                continue

            out_json = vectors_path / f"{file_path.stem}_vector.json"
            if out_json.exists():
                processed_count += 1
                continue

            try:
                raw_vector = self._vectorizer.get_embedding(file_path)
                embedding = VectorEmbedding(values=tuple(raw_vector), model_name="Qwen-VL")
                normalized = embedding.l2_normalize()

                with open(out_json, "w") as f:
                    json.dump({
                        "file": file_path.name,
                        "model": normalized.model_name,
                        "vector": list(normalized.values)
                    }, f)

                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to vectorize {file_path.name}: {e}")

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Vectorized {processed_count} images."
        )