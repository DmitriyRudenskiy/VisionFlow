from src.application.ports import ImageEmbeddingExtractorPort


class QwenVLClient(ImageEmbeddingExtractorPort):
    def get_embedding(self, image_path):
        return [0.0] * 512