# src/application/pipeline/steps/pose_extraction_step.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.application.pipeline.steps.batch_file_step import BatchFileProcessingStep
from src.application.ports import StoragePort, PoseExtractionPort
from src.domain.image.value_objects import FilePath
from src.domain.metadata.entities import PoseData
from src.domain.metadata.value_objects import PoseKeypoint


class PoseExtractionStep(BatchFileProcessingStep):
    def __init__(self, storage: StoragePort, extractor: Optional[PoseExtractionPort] = None):
        super().__init__(storage)
        self._extractor_factory = extractor
        self._extractor: Optional[PoseExtractionPort] = None

    def prepare(self) -> None:
        if self._extractor is None:
            if self._extractor_factory is not None:
                self._extractor = self._extractor_factory
            else:
                from src.infrastructure.ai.dwpose_client import DWPoseClient
                self._extractor = DWPoseClient()

    @property
    def output_subdirectory(self) -> str:
        return "_poses"

    @property
    def output_suffix(self) -> str:
        return "_pose.json"

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        if self._extractor is None:
            raise RuntimeError("prepare() was not called before execute()")

        raw_kp = self._extractor.extract_keypoints(file_path)
        pose_data = PoseData(
            file_path=FilePath(path=file_path),
            keypoints_body=self._map_keypoints(raw_kp.get("body", [])),
            keypoints_face=self._map_keypoints(raw_kp.get("face", [])),
            keypoints_left_hand=self._map_keypoints(raw_kp.get("left_hand", [])),
            keypoints_right_hand=self._map_keypoints(raw_kp.get("right_hand", [])),
        )
        return {
            "file": file_path.name,
            "total_keypoints": pose_data.total_keypoints,
            "body": raw_kp.get("body", []),
        }

    def _map_keypoints(self, kp_list: list) -> list:
        return [
            PoseKeypoint(x=kp["x"], y=kp["y"], confidence=kp["confidence"], name=kp["name"])
            for kp in kp_list
        ]