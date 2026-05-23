# tests/test_pipeline_steps.py
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

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
from src.infrastructure.ai.vit_client import VisionTransformerClient
from src.infrastructure.ai.sam3_client import SAM3Client
from src.infrastructure.ai.qwen_client import QwenVLClient
from src.infrastructure.ai.dwpose_client import DWPoseClient
from src.infrastructure.ai.nsfw_client import NSFWClient
from src.infrastructure.ai.color_client import ColorExtractorClient


@pytest.fixture
def fs():
    return FileSystemService()


# ------------------------------------------------------------------ helpers
def make_step_config(step_number: int, source: Path) -> StepConfigDTO:
    return StepConfigDTO(
        step_number=step_number,
        params={"source_path": str(source), "output_path": str(source)},
    )


def create_test_file(path: Path, name: str, content: bytes = b"img") -> Path:
    """Create a minimal 'image' file with a valid extension."""
    file_path = path / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)
    return file_path


# ------------------------------------------------------------------ tests
class TestStep0Flatten:
    def test_flatten_moves_files_to_root(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        # nested file
        nested = src / "sub"
        nested.mkdir()
        create_test_file(nested, "a.jpg")
        # file already in root
        create_test_file(src, "b.png")
        # duplicate name in root and sub
        create_test_file(src, "c.jpg")
        create_test_file(nested, "c.jpg")

        step = Step0Flatten(FileSystemService())
        config = make_step_config(0, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert result.message.startswith("Flattened directory")
        # all files should be in root
        root_files = list(src.iterdir())
        assert len([f for f in root_files if f.is_file()]) == 4
        assert (src / "a.jpg").exists()
        assert (src / "b.png").exists()
        assert (src / "c.jpg").exists()
        # duplicated should have been renamed
        assert any(f.name.startswith("c_") for f in root_files if f.is_file())


class TestStep1Prepare:
    def test_backup_and_rename(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "photo.jpg")
        create_test_file(src, "image.png")
        # non‑image file should be ignored
        (src / "readme.txt").write_text("hello")

        step = Step1Prepare(FileSystemService())
        config = make_step_config(1, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert (src / "_originals").is_dir()
        # originals copied
        assert (src / "_originals" / "photo.jpg").exists()
        assert (src / "_originals" / "image.png").exists()
        # originals folder should not contain timestamped files
        assert len(list((src / "_originals").iterdir())) == 2
        # original files should be renamed with timestamp
        renamed = [f for f in src.iterdir() if f.is_file()]
        assert len(renamed) == 3  # two images + one txt unchanged
        for f in renamed:
            if f.suffix.lower() in (".jpg", ".png"):
                assert "_" in f.stem  # timestamp format


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
        # one of the duplicates should be moved
        duplicates = list(dup_dir.iterdir())
        assert len(duplicates) == 1
        # original should remain
        assert (src / "orig.jpg").exists() or (src / "copy.jpg").exists()
        # unique file untouched
        assert (src / "unique.png").exists()


class TestStep3VisualDups:
    def test_visual_duplicates_creates_report(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "a.jpg")
        create_test_file(src, "b.png")

        # mock phash to return different values -> no duplicates
        detector = VisionTransformerClient()
        step = Step3VisualDups(FileSystemService(), detector)
        config = make_step_config(3, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert (src / "_visual_duplicates").is_dir()
        report = src / "visual_dups_report.html"
        assert report.exists()
        assert "Visual Duplicates Report" in report.read_text()


class TestStep4AICrop:
    def test_ai_crop_moves_files(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "pic.jpg")
        create_test_file(src, "img.png")

        segmenter = SAM3Client()
        step = Step4AICrop(FileSystemService(), segmenter)
        config = make_step_config(4, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        crop_dir = src / "_ai_cropped"
        assert crop_dir.is_dir()
        assert len(list(crop_dir.iterdir())) == 2


class TestStep5Vectorize:
    def test_creates_vector_json(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "img.jpg")

        vectorizer = QwenVLClient()
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

        pose_ext = DWPoseClient()
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

        color_ext = ColorExtractorClient()
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

        nsfw_clf = NSFWClient()
        step = Step8NsfwScore(FileSystemService(), nsfw_clf)
        config = make_step_config(8, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        nsfw_dir = src / "_nsfw"
        assert nsfw_dir.is_dir()
        json_files = list(nsfw_dir.glob("*_nsfw.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["safe"] == 1.0