# infrastructure/ai/qwen_client.py
from src.application.ports import VectorizationPort

class QwenVLClient(VectorizationPort):
    def get_embedding(self, image_path):
        # Заглушка: возвращает список из 512 нулей
        return [0.0] * 512