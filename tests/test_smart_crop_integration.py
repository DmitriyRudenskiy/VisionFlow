# tests/test_smart_crop_integration.py
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.application.pipeline.dto import StepConfigDTO
from src.application.pipeline.steps.smart_crop_step import SmartCropStep
from src.infrastructure.file_system import FileSystemStorage

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "crop"


@pytest.fixture
def file_storage() -> FileSystemStorage:
    return FileSystemStorage()


def _is_original(path: Path) -> bool:
    """Проверяет, что файл — оригинал, а не кроп (_person_)."""
    return "_person_" not in path.name


@pytest.fixture
def crop_fixture_dir(tmp_path: Path) -> Path:
    """Копирует оригинальные изображения (любой формат, без _person_) из fixtures."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Fixtures directory not found: {FIXTURES_DIR}")

    target = tmp_path / "source"
    target.mkdir(parents=True, exist_ok=True)

    originals = [
        p for p in FIXTURES_DIR.glob("*")
        if p.is_file() and _is_original(p)
    ]
    if not originals:
        pytest.skip(f"No original images found in {FIXTURES_DIR}")

    for img in originals:
        shutil.copy(img, target / img.name)

    return target


@pytest.fixture
def expected_crops_by_original() -> dict[str, list[Path]]:
    """Сопоставляет stem оригинала с эталонными кропами."""
    if not FIXTURES_DIR.exists():
        return {}

    mapping: dict[str, list[Path]] = {}
    for original in FIXTURES_DIR.glob("*"):
        if not original.is_file() or not _is_original(original):
            continue
        stem = original.stem
        crops = sorted(FIXTURES_DIR.glob(f"{stem}_person_*.jpg"))
        if crops:
            mapping[stem] = crops

    return mapping


class TestSmartCropIntegration:
    """Интеграционный тест SmartCropStep с эталонными изображениями."""

    def test_smart_crop_generates_expected_files(
        self,
        crop_fixture_dir: Path,
        expected_crops_by_original: dict[str, list[Path]],
        file_storage: FileSystemStorage,
    ) -> None:
        """Проверяет, что шаг корректно именует и сохраняет кропы, не трогая оригиналы."""
        if not expected_crops_by_original:
            pytest.skip("No expected crops found in fixtures")

        output_dir = crop_fixture_dir.parent / "output"
        output_dir.mkdir()

        temp_dir = crop_fixture_dir.parent / "temp_crops"
        temp_dir.mkdir()

        segmenter = MagicMock()

        def mock_crop(image_path: Path, mode: str = "square") -> list[Path]:
            stem = image_path.stem
            expected = expected_crops_by_original.get(stem, [])
            temp_crops: list[Path] = []
            for crop_path in expected:
                tmp_crop = temp_dir / f"tmp_{stem}_{crop_path.name}"
                shutil.copy(crop_path, tmp_crop)
                temp_crops.append(tmp_crop)
            return temp_crops

        segmenter.crop_image.side_effect = mock_crop

        step = SmartCropStep(file_storage, segmenter=segmenter)
        step.prepare()

        config = StepConfigDTO(
            sequence_number=4,
            params={
                "source_path": str(crop_fixture_dir),
                "output_path": str(output_dir),
                "crop_mode": "square",
            },
        )

        result = step.execute(config)

        assert result.status == "COMPLETED"
        total_expected = sum(len(crops) for crops in expected_crops_by_original.values())
        assert result.processed_count == total_expected

        # Проверяем имена и содержимое каждого кропа
        for stem, expected_crops in expected_crops_by_original.items():
            for idx, expected_path in enumerate(expected_crops, start=1):
                expected_name = f"{stem}_person_{idx}{expected_path.suffix}"
                actual_path = output_dir / expected_name
                assert actual_path.exists(), f"Expected output file not found: {expected_name}"
                assert actual_path.read_bytes() == expected_path.read_bytes(), (
                    f"Content mismatch for {expected_name}"
                )

        # Оригиналы остались нетронутыми (любой графический формат)
        for original in crop_fixture_dir.glob("*"):
            if original.is_file() and _is_original(original):
                assert original.exists(), f"Original file was modified: {original.name}"

        # Временные файлы подчищены
        remaining_temps = list(temp_dir.iterdir())
        assert len(remaining_temps) == 0, f"Temp files not cleaned up: {remaining_temps}"

    def test_smart_crop_empty_segmentation(
        self,
        crop_fixture_dir: Path,
        file_storage: FileSystemStorage,
    ) -> None:
        """Если сегментатор ничего не вернул — output пуст, ошибок нет, оригиналы на месте."""
        output_dir = crop_fixture_dir.parent / "output_empty"
        output_dir.mkdir()

        segmenter = MagicMock()
        segmenter.crop_image.return_value = []

        step = SmartCropStep(file_storage, segmenter=segmenter)
        step.prepare()

        config = StepConfigDTO(
            sequence_number=4,
            params={
                "source_path": str(crop_fixture_dir),
                "output_path": str(output_dir),
                "crop_mode": "square",
            },
        )

        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert result.processed_count == 0
        assert len(list(output_dir.iterdir())) == 0

        for original in crop_fixture_dir.glob("*"):
            if original.is_file() and _is_original(original):
                assert original.exists()