from src.application.ports import ColorPaletteExtractorPort


class ColorExtractorClient(ColorPaletteExtractorPort):
    def extract_palette(self, image_path, num_colors=5):
        return [{"rgb": [128, 128, 128], "hex": "#808080", "percentage": 100.0}]