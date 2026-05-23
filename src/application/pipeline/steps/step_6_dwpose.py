# application/pipeline/steps/step_6_dwpose.py
import json
import logging
from pathlib import Path
from typing import List

from src.application.pipeline.steps.base_step import BaseStep
from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.ports import FileSystemServicePort, PoseExtractorPort
from src.domain.image.value_objects import FilePath
from src.domain.metadata.entities import PoseData
from src.domain.metadata.value_objects import PoseKeypoint

logger = logging.getLogger(__name__)


class Step6DWPose(BaseStep):
    def __init__(
        self, fs_service: FileSystemServicePort, pose_extractor: PoseExtractorPort
    ):
        self._fs = fs_service
        self._pose_extractor = pose_extractor

    def _map_keypoints(self, kp_list: list) -> List[PoseKeypoint]:
        return [
            PoseKeypoint(
                x=kp["x"], y=kp["y"], confidence=kp["confidence"], name=kp["name"]
            )
            for kp in kp_list
        ]

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        source_path = Path(config.params.get("source_path", "."))
        poses_path = source_path / "_poses"
        self._fs.create_directory(poses_path)

        all_files = self._fs.scan_directory(source_path, recursive=False)
        processed_count = 0

        for file_path in all_files:
            if not file_path.is_file():
                continue

            out_json = poses_path / f"{file_path.stem}_pose.json"

            if out_json.exists():
                processed_count += 1
                continue

            try:
                raw_kp = self._pose_extractor.extract_keypoints(file_path)
                pose_data = PoseData(
                    file_path=FilePath(path=file_path),
                    keypoints_body=self._map_keypoints(raw_kp.get("body", [])),
                    keypoints_face=self._map_keypoints(raw_kp.get("face", [])),
                    keypoints_left_hand=self._map_keypoints(
                        raw_kp.get("left_hand", [])
                    ),
                    keypoints_right_hand=self._map_keypoints(
                        raw_kp.get("right_hand", [])
                    ),
                )

                with open(out_json, "w") as f:
                    json.dump(
                        {
                            "file": file_path.name,
                            "total_keypoints": pose_data.total_keypoints,
                            "body": raw_kp.get("body", []),
                        },
                        f,
                    )

                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to extract pose {file_path.name}: {e}")

        return StepResultDTO(
            step_number=config.step_number,
            status="COMPLETED",
            message=f"Extracted poses for {processed_count} images.",
            processed_count=processed_count,
        )
