# infrastructure/ai/nsfw_client.py
from src.application.ports import NsfwClassifierPort

class NSFWClient(NsfwClassifierPort):
    def classify(self, image_path):
        # Заглушка: всегда 0.0 / 1.0
        return (0.0, 1.0)