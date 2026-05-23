# infrastructure/ai/vit_client.py
from src.application.ports import VisualDupDetectorPort

class VisionTransformerClient(VisualDupDetectorPort):
    def calculate_phash(self, image_path):
        # Заглушка
        return "phash_placeholder"

    def calculate_vit_similarity(self, image_path1, image_path2):
        # Заглушка
        return 0.0