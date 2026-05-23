# infrastructure/ai/dwpose_client.py
from src.application.ports import PoseExtractorPort

class DWPoseClient(PoseExtractorPort):
    def extract_keypoints(self, image_path):
        # Заглушка: возвращает пустые ключевые точки
        return {"body": [], "face": [], "left_hand": [], "right_hand": []}