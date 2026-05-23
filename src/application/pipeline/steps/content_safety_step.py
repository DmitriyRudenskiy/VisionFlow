# src/application/pipeline/steps/content_safety_step.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.application.pipeline.steps.batch_file_step import BatchFileProcessingStep
from src.application.ports import StoragePort, ContentSafetyClassifierPort
from src.domain.metadata.value_objects import NsfwScore


class ContentSafetyClassificationStep(BatchFileProcessingStep):
    def __init__(self, storage: StoragePort, classifier: Optional[ContentSafetyClassifierPort] = None):
        super().__init__(storage)
        self._classifier_factory = classifier
        self._classifier: Optional[ContentSafetyClassifierPort] = None

    def prepare(self) -> None:
        if self._classifier is None:
            if self._classifier_factory is not None:
                self._classifier = self._classifier_factory
            else:
                from src.infrastructure.ai.nsfw_client import NSFWClient
                self._classifier = NSFWClient()

    @property
    def output_subdirectory(self) -> str:
        return "_nsfw"

    @property
    def output_suffix(self) -> str:
        return "_nsfw.json"

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        if self._classifier is None:
            raise RuntimeError("prepare() was not called before execute()")

        nsfw_val, safe_val = self._classifier.classify(file_path)
        score = NsfwScore(nsfw_value=nsfw_val, safe_value=safe_val)
        return {
            "file": file_path.name,
            "nsfw": score.nsfw_value,
            "safe": score.safe_value,
        }