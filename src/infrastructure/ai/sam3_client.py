# infrastructure/ai/sam3_client.py
from src.application.ports import AISegmenterPort

class SAM3Client(AISegmenterPort):
    def crop_image(self, image_path, mode="square"):
        # Заглушка: просто возвращает тот же путь
        return image_path