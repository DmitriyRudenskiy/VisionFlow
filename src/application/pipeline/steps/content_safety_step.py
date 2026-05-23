from pathlib import Path
from typing import Any, Dict

from src.application.pipeline.steps.batch_file_step import BatchFileProcessingStep
from src.application.ports import StoragePort, ContentSafetyClassifierPort
from src.domain.metadata.value_objects import NsfwScore


class ContentSafetyClassificationStep(BatchFileProcessingStep):
    def __init__(self, storage: StoragePort, classifier: ContentSafetyClassifierPort):
        super().__init__(storage)
        self._classifier = classifier

    @property
    def output_subdirectory(self) -> str:
        return "_nsfw"

    @property
    def output_suffix(self) -> str:
        return "_nsfw.json"

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        nsfw_val, safe_val = self._classifier.classify(file_path)
        score = NsfwScore(nsfw_value=nsfw_val, safe_value=safe_val)
        return {
            "file": file_path.name,
            "nsfw": score.nsfw_value,
            "safe": score.safe_value,
        }