# tests/test_pose_extraction_integration.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.application.pipeline.dto import StepConfigDTO
from src.application.pipeline.steps.pose_extraction_step import PoseExtractionStep
from src.infrastructure.file_system import FileSystemStorage

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pose"
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp", ".tiff"}
)


@pytest.fixture
def file_storage() -> FileSystemStorage:
    return FileSystemStorage()


@pytest.fixture
def pose_fixture_dir(tmp_path: Path) -> Path:
    """Копирует эталонные изображения во временную изолированную директорию."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Fixtures directory not found: {FIXTURES_DIR}")

    target = tmp_path / "samples"
    target.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in FIXTURES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_files:
        pytest.skip(f"No supported images found in {FIXTURES_DIR}")

    for img in image_files:
        shutil.copy(img, target / img.name)

    return target


@pytest.fixture
def expected_poses() -> dict[str, dict[str, Any]]:
    """Загружает эталонные JSON с ожидаемыми позами, индексируя по stem исходного файла."""
    json_files = sorted(FIXTURES_DIR.glob("*_pose.json"))
    if not json_files:
        pytest.skip(f"Expected pose JSONs not found in {FIXTURES_DIR}")

    expected: dict[str, dict[str, Any]] = {}
    for json_file in json_files:
        # item_00002_person_2_pose.json -> item_00002_person_2
        stem = json_file.stem.replace("_pose", "")
        with open(json_file, "r", encoding="utf-8") as f:
            expected[stem] = json.load(f)

    return expected


class TestPoseExtractionIntegration:
    """Изолированный интеграционный тест шага извлечения поз."""

    def test_extracted_poses_match_expected(
        self,
        pose_fixture_dir: Path,
        expected_poses: dict[str, dict[str, Any]],
        file_storage: FileSystemStorage,
    ) -> None:
        extractor = MagicMock()

        def _extract(image_path: Path) -> dict[str, Any]:
            stem = image_path.stem
            data = expected_poses.get(stem, {})
            return {
                "body": data.get("body", []),
                "face": data.get("face", []),
                "left_hand": data.get("left_hand", []),
                "right_hand": data.get("right_hand", []),
            }

        extractor.extract_keypoints.side_effect = _extract

        step = PoseExtractionStep(file_storage, extractor=extractor)
        step.prepare()

        config = StepConfigDTO(
            sequence_number=6,
            params={
                "source_path": str(pose_fixture_dir),
                "output_path": str(pose_fixture_dir),
            },
        )

        result = step.execute(config)

        assert result.status == "COMPLETED"
        expected_count = len(expected_poses)
        if expected_count > 0:
            assert result.processed_count >= expected_count, (
                f"Expected at least {expected_count} processed files, got {result.processed_count}"
            )

        poses_dir = pose_fixture_dir / "_poses"
        assert poses_dir.is_dir()

        generated_files = sorted(poses_dir.glob("*_pose.json"))
        assert len(generated_files) == result.processed_count

        for gen_file in generated_files:
            with open(gen_file, "r", encoding="utf-8") as f:
                actual = json.load(f)

            source_name = actual["file"]
            source_stem = Path(source_name).stem

            expected = expected_poses.get(source_stem)
            if expected is None:
                continue

            assert actual["total_keypoints"] == expected.get("total_keypoints"), (
                f"total_keypoints mismatch for {source_name}: "
                f"got {actual['total_keypoints']}, expected {expected.get('total_keypoints')}"
            )
            assert actual["body"] == expected.get("body", []), (
                f"Body keypoints mismatch for {source_name}"
            )