#!/usr/bin/env python3
"""Verify that all refactored modules can be imported without errors."""

import sys


def verify():
    errors = []

    try:
        from src.domain.pipeline.entities import PipelineAggregate, PipelineStep
        from src.domain.pipeline.value_objects import PipelineStatus, StepStatus
    except Exception as e:
        errors.append(f"Domain pipeline imports failed: {e}")

    try:
        from src.application.pipeline.dto import PipelineConfigDTO, StepConfigDTO, StepResultDTO
        from src.application.pipeline.orchestrator import PipelineOrchestrator
        from src.application.pipeline.step_registry import StepRegistry
        from src.application.pipeline.steps.base_step import BaseStep
        from src.application.pipeline.steps.batch_file_step import BatchFileProcessingStep
        from src.application.pipeline.steps.flatten_directories_step import FlattenDirectoriesStep
        from src.application.pipeline.steps.prepare_images_step import PrepareImagesStep
        from src.application.pipeline.steps.exact_deduplication_step import ExactDeduplicationStep
        from src.application.pipeline.steps.visual_deduplication_step import VisualDeduplicationStep
        from src.application.pipeline.steps.smart_crop_step import SmartCropStep
        from src.application.pipeline.steps.embedding_extraction_step import EmbeddingExtractionStep
        from src.application.pipeline.steps.pose_extraction_step import PoseExtractionStep
        from src.application.pipeline.steps.color_palette_extraction_step import ColorPaletteExtractionStep
        from src.application.pipeline.steps.content_safety_step import ContentSafetyClassificationStep
    except Exception as e:
        errors.append(f"Application pipeline imports failed: {e}")

    try:
        from src.infrastructure.file_system import FileSystemStorage
        from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
        from src.infrastructure.persistence.pipeline_serializer import JsonPipelineSerializer, PipelineSerializer
    except Exception as e:
        errors.append(f"Infrastructure imports failed: {e}")

    try:
        from src.infrastructure.ai.vit_client import VisionTransformerClient
        from src.infrastructure.ai.sam3_client import SAM3Client
        from src.infrastructure.ai.qwen_client import QwenVLClient
        from src.infrastructure.ai.dwpose_client import DWPoseClient
        from src.infrastructure.ai.nsfw_client import NSFWClient
        from src.infrastructure.ai.color_client import ColorExtractorClient
    except Exception as e:
        errors.append(f"AI client imports failed: {e}")

    if errors:
        print("❌ Import verification failed:")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)

    print("✅ All module imports resolved successfully.")
    print("   Verified domains: Pipeline, Application, Infrastructure")
    sys.exit(0)


if __name__ == "__main__":
    verify()