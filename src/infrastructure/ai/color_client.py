# infrastructure/ai/color_client.py
from src.application.ports import ColorExtractorPort


class ColorExtractorClient(ColorExtractorPort):
    def extract_palette(self, image_path, num_colors=5):
        # Заглушка: возвращает фиктивную палитру
        return [{"rgb": [128, 128, 128], "hex": "#808080", "percentage": 100.0}]
