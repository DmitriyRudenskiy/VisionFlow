# src/infrastructure/ai/vit_client.py
from src.application.ports import VisualDupDetectorPort
import uuid


class VisionTransformerClient(VisualDupDetectorPort):
    def calculate_phash(self, image_path):
        return f"phash_{uuid.uuid4().hex[:8]}"

    def calculate_vit_similarity(self, image_path1, image_path2):
        return 0.0
