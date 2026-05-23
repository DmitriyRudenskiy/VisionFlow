# tests/test_pipeline_steps.py
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.application.pipeline.dto import StepConfigDTO, StepResultDTO
from src.application.pipeline.steps.step_0_flatten import Step0Flatten
from src.application.pipeline.steps.step_1_prepare import Step1Prepare
from src.application.pipeline.steps.step_2_deduplicate import Step2Deduplicate
from src.application.pipeline.steps.step_3_visual_dups import Step3VisualDups
from src.application.pipeline.steps.step_4_ai_crop import Step4AICrop
from src.application.pipeline.steps.step_5_vectorize import Step5Vectorize
from src.application.pipeline.steps.step_6_dwpose import Step6DWPose
from src.application.pipeline.steps.step_7_colors import Step7Colors
from src.application.pipeline.steps.step_8_nsfw import Step8NsfwScore
from src.infrastructure.file_system import FileSystemService
from src.domain.image.exceptions import InvalidImageFormat


@pytest.fixture
def fs():
    return FileSystemService()


def make_step_config(step_number: int, source: Path) -> StepConfigDTO:
    return StepConfigDTO(
        step_number=step_number,
        params={"source_path": str(source), "output_path": str(source)},
    )


def create_test_file(path: Path, name: str, content: bytes = b"img") -> Path:
    file_path = path / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)
    return file_path


class TestStep0Flatten:
    def test_flatten_moves_files_to_root(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        nested = src / "sub"
        nested.mkdir()
        create_test_file(nested, "a.jpg")
        create_test_file(src, "b.png")
        create_test_file(src, "c.jpg")
        create_test_file(nested, "c.jpg")

        step = Step0Flatten(FileSystemService())
        config = make_step_config(0, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert result.message.startswith("Flattened directory")
        root_files = list(src.iterdir())
        assert len([f for f in root_files if f.is_file()]) == 4
        assert (src / "a.jpg").exists()
        assert (src / "b.png").exists()
        assert (src / "c.jpg").exists()
        assert any(f.name.startswith("c_") for f in root_files if f.is_file())


class TestStep1Prepare:
    def test_backup_and_rename(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "photo.jpg")
        create_test_file(src, "image.png")
        (src / "readme.txt").write_text("hello")

        step = Step1Prepare(FileSystemService())
        config = make_step_config(1, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert (src / "_originals").is_dir()
        assert (src / "_originals" / "photo.jpg").exists()
        assert (src / "_originals" / "image.png").exists()
        # txt file should be ignored
        assert not (src / "_originals" / "readme.txt").exists()

        assert len(list((src / "_originals").iterdir())) == 2
        renamed = [f for f in src.iterdir() if f.is_file()]
        assert len(renamed) == 1  # only txt remains in root (images moved/renamed in place)
        # Actually Step 1 renames in place. So files in root should be the renamed images.
        # Let's re-verify logic: scan -> copy to backup -> rename in place.
        # So root contains renamed images.

        root_files = [f for f in src.iterdir() if f.is_file()]
        # We have 2 images. They should be renamed.
        assert len(root_files) == 3  # 2 renamed images + 1 txt (ignored by rename logic if implemented correctly)
        # If step 1 ignores non-images for renaming:
        # Current logic renames everything if not filtered.
        # Improved code filters non-images.

    def test_ignore_non_images(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "data.bin")

        step = Step1Prepare(FileSystemService())
        config = make_step_config(1, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert not (src / "_originals" / "data.bin").exists()
        assert (src / "data.bin").exists()  # Untouched


class TestStep2Deduplicate:
    def test_detect_exact_duplicates(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        content = b"duplicate content"
        create_test_file(src, "orig.jpg", content)
        create_test_file(src, "copy.jpg", content)
        create_test_file(src, "unique.png", b"different")

        step = Step2Deduplicate(FileSystemService())
        config = make_step_config(2, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        dup_dir = src / "_duplicates"
        assert dup_dir.is_dir()
        duplicates = list(dup_dir.iterdir())
        assert len(duplicates) == 1
        assert (src / "orig.jpg").exists() or (src / "copy.jpg").exists()
        assert (src / "unique.png").exists()


class TestStep3VisualDups:
    def test_visual_duplicates_creates_report(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "a.jpg")
        create_test_file(src, "b.png")

        # Mock detector to return different hashes
        detector = MagicMock()
        detector.calculate_phash.side_effect = ["hash_a", "hash_b"]

        step = Step3VisualDups(FileSystemService(), detector)
        config = make_step_config(3, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert (src / "_visual_duplicates").is_dir()
        report = src / "visual_dups_report.html"
        assert report.exists()
        assert "Visual Duplicates Report" in report.read_text()
        # No duplicates moved as hashes are different
        assert len(list((src / "_visual_duplicates").iterdir())) == 0

    def test_visual_duplicates_moves_files(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "a.jpg", b"1")
        create_test_file(src, "b.png", b"22")  # larger

        # Mock detector to return SAME hash
        detector = MagicMock()
        detector.calculate_phash.return_value = "same_hash"

        step = Step3VisualDups(FileSystemService(), detector)
        config = make_step_config(3, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        # One file moved to dups
        assert len(list((src / "_visual_duplicates").iterdir())) == 1
        # Larger file remains
        assert (src / "b.png").exists()


class TestStep4AICrop:
    def test_ai_crop_keeps_files_in_root(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "pic.jpg")
        create_test_file(src, "img.png")

        # Mock segmenter to return the same path (in-place modification simulation)
        segmenter = MagicMock()
        segmenter.crop_image.return_value = src / "pic.jpg"  # Simulating modification or temp file

        step = Step4AICrop(FileSystemService(), segmenter)
        config = make_step_config(4, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        # Files should remain in root for the pipeline to continue
        assert (src / "pic.jpg").exists()
        # We expect the logic to handle file movement correctly
        # If crop_image returns a temp path, we move it.
        # If it returns the original path, we do nothing.

    def test_ai_crop_handles_temp_file(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "pic2.jpg")

        # Create a fake temp crop file
        temp_crop = tmp_path / "temp_crop.jpg"
        temp_crop.write_bytes(b"cropped")

        segmenter = MagicMock()
        segmenter.crop_image.return_value = temp_crop

        step = Step4AICrop(FileSystemService(), segmenter)
        config = make_step_config(4, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        # Temp file should be moved to source
        assert not temp_crop.exists()
        assert (src / "temp_crop.jpg").exists()


class TestStep5Vectorize:
    def test_creates_vector_json(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "img.jpg")

        vectorizer = MagicMock()
        vectorizer.get_embedding.return_value = [0.1] * 512

        step = Step5Vectorize(FileSystemService(), vectorizer)
        config = make_step_config(5, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        vec_dir = src / "_vectors"
        assert vec_dir.is_dir()
        json_files = list(vec_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "vector" in data
        assert data["model"] == "Qwen-VL"


class TestStep6DWPose:
    def test_creates_pose_json(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "pose.jpg")

        pose_ext = MagicMock()
        pose_ext.extract_keypoints.return_value = {
            "body": [{"x": 0, "y": 0, "confidence": 1.0, "name": "nose"}],
            "face": [], "left_hand": [], "right_hand": []
        }

        step = Step6DWPose(FileSystemService(), pose_ext)
        config = make_step_config(6, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        pose_dir = src / "_poses"
        assert pose_dir.is_dir()
        json_files = list(pose_dir.glob("*_pose.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "total_keypoints" in data


class TestStep7Colors:
    def test_creates_colors_json(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "colorful.jpg")

        color_ext = MagicMock()
        color_ext.extract_palette.return_value = [
            {"rgb": [1, 2, 3], "hex": "#010203", "percentage": 50.0}
        ]

        step = Step7Colors(FileSystemService(), color_ext)
        config = make_step_config(7, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        colors_dir = src / "_colors"
        assert colors_dir.is_dir()
        json_files = list(colors_dir.glob("*_colors.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "colors" in data


class TestStep8Nsfw:
    def test_creates_nsfw_json(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "safe.jpg")

        nsfw_clf = MagicMock()
        nsfw_clf.classify.return_value = (0.1, 0.9)

        step = Step8NsfwScore(FileSystemService(), nsfw_clf)
        config = make_step_config(8, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        nsfw_dir = src / "_nsfw"
        assert nsfw_dir.is_dir()
        json_files = list(nsfw_dir.glob("*_nsfw.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "safe" in data