# src/infrastructure/ai/sam3_client.py
from src.application.ports import AISegmenterPort


class SAM3Client(AISegmenterPort):
    def crop_image(self, image_path, mode="square"):
        # Simulate creating a new file in a temp location or modifying in place
        # Here we simulate returning the original path (in-place modification)
        return image_path
