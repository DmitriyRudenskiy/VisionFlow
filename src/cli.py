# src/cli.py
import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID

from src.application.pipeline.orchestrator import PipelineOrchestrator
from src.application.pipeline.step_registry import StepRegistry
from src.application.pipeline.dto import PipelineConfigDTO
from src.infrastructure.file_system import FileSystemService
from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
from src.infrastructure.ai.vit_client import VisionTransformerClient
from src.infrastructure.ai.sam3_client import SAM3Client
from src.infrastructure.ai.qwen_client import QwenVLClient
from src.infrastructure.ai.dwpose_client import DWPoseClient
from src.infrastructure.ai.nsfw_client import NSFWClient
from src.infrastructure.ai.color_client import ColorExtractorClient
from src.application.pipeline.steps.step_0_flatten import Step0Flatten
from src.application.pipeline.steps.step_1_prepare import Step1Prepare
from src.application.pipeline.steps.step_2_deduplicate import Step2Deduplicate
from src.application.pipeline.steps.step_3_visual_dups import Step3VisualDups
from src.application.pipeline.steps.step_4_ai_crop import Step4AICrop
from src.application.pipeline.steps.step_5_vectorize import Step5Vectorize
from src.application.pipeline.steps.step_6_dwpose import Step6DWPose
from src.application.pipeline.steps.step_7_colors import Step7Colors
from src.application.pipeline.steps.step_8_nsfw import Step8NsfwScore


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)


def build_container() -> PipelineOrchestrator:
    repo = JsonPipelineRepository(storage_dir=Path("./data/pipelines"))
    fs_service = FileSystemService()
    logger.info("Initializing AI Clients...")
    visual_detector = VisionTransformerClient()
    ai_segmenter = SAM3Client()
    vectorizer = QwenVLClient()
    pose_extractor = DWPoseClient()
    nsfw_classifier = NSFWClient()
    color_extractor = ColorExtractorClient()

    registry = StepRegistry()
    registry.register(0, Step0Flatten(fs_service))
    registry.register(1, Step1Prepare(fs_service))
    registry.register(2, Step2Deduplicate(fs_service))
    registry.register(3, Step3VisualDups(fs_service, visual_detector))
    registry.register(4, Step4AICrop(fs_service, ai_segmenter))
    registry.register(5, Step5Vectorize(fs_service, vectorizer))
    registry.register(6, Step6DWPose(fs_service, pose_extractor))
    registry.register(7, Step7Colors(fs_service, color_extractor))
    registry.register(8, Step8NsfwScore(fs_service, nsfw_classifier))

    return PipelineOrchestrator(registry, repo)


def main():
    parser = argparse.ArgumentParser(
        description="VisionFlow: Intelligent Image Processing Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli ./input_images ./output_dir
  python -m src.cli '/Users/user/Downloads/Новая папка' '/Users/user/Downloads/Новая папка' --steps 0
  python -m src.cli ./input_images ./output_dir --pipeline-id 123e4567-e89b-12d3-a456-426614174000
        """
    )
    parser.add_argument("source", type=Path, help="Путь к директории с исходными изображениями")
    parser.add_argument("output", type=Path, help="Путь к выходной директории")
    parser.add_argument("--steps", nargs="+", type=int, help="Список номеров шагов")
    parser.add_argument("--pipeline-id", type=UUID, help="UUID существующего пайплайна")
    parser.add_argument("--no-stop-on-error", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.source.exists():
        logger.error(f"Исходная директория не найдена: {args.source}")
        sys.exit(1)
    if not args.source.is_dir():
        logger.error(f"Указанный путь не является директорией: {args.source}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    orchestrator = build_container()
    available_steps = orchestrator.get_available_steps()
    if args.steps:
        invalid_steps = set(args.steps) - set(available_steps)
        if invalid_steps:
            logger.error(f"Неизвестные шаги: {invalid_steps}. Доступные: {available_steps}")
            sys.exit(1)

    config = PipelineConfigDTO(
        source_path=args.source.resolve(),
        output_path=args.output.resolve(),
        steps_to_run=args.steps,
        stop_on_error=not args.no_stop_on_error
    )

    logger.info("Запуск VisionFlow Pipeline...")
    logger.info(f"Source: {config.source_path}")
    logger.info(f"Output: {config.output_path}")
    if args.pipeline_id:
        logger.info(f"Resume Mode: ID={args.pipeline_id}")

    try:
        results = list(orchestrator.execute(config, pipeline_id=args.pipeline_id))
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        success_count = 0
        fail_count = 0
        for res in results:
            icon = "✅" if res.status == "COMPLETED" else "❌"
            msg = f"{icon} Step {res.step_number}: {res.status}"
            if res.message:
                msg += f" | {res.message}"
            if res.status == "FAILED":
                logger.warning(msg)
                if res.errors:
                    for err in res.errors:
                        logger.error(f"   └── Error: {err}")
                fail_count += 1
            else:
                logger.info(msg)
                success_count += 1
        logger.info("-" * 50)
        logger.info(f"Total: {success_count} succeeded, {fail_count} failed.")
        logger.info("=" * 50)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Critical unhandled error during pipeline execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()