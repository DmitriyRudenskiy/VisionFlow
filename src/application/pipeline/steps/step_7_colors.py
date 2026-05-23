# application/pipeline/steps/step_8_nsfw.py
import json
import logging
from pathlib import Path
from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort, NsfwClassifierPort
from src.domain.image.value_objects import FilePath
from src.domain.metadata.value_objects import NsfwScore

logger = logging.getLogger(__name__)


class Step8NsfwScore(BaseStep):
    def __init__(self, fs_service: FileSystemServicePort, nsfw_classifier: NsfwClassifierPort):
        self._fs = fs_service
        self._nsfw_classifier = nsfw_classifier

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        nsfw_path = source_path / "_nsfw"
        self._fs.create_directory(nsfw_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)
        processed_count = 0

        for file_path in all_files:
            if not file_path.is_file():
                continue

            out_json = nsfw_path / f"{file_path.stem}_nsfw.json"
            if out_json.exists():
                processed_count += 1
                continue

            try:
                nsfw_val, safe_val = self._nsfw_classifier.classify(file_path)
                score = NsfwScore(nsfw_value=nsfw_val, safe_value=safe_val)

                with open(out_json, "w") as f:
                    json.dump({
                        "file": file_path.name,
                        "nsfw": score.nsfw_value,
                        "safe": score.safe_value
                    }, f)

                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to classify NSFW {file_path.name}: {e}")

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Classified NSFW scores for {processed_count} images.",
            processed_count=processed_count
        )