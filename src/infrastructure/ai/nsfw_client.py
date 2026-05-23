from src.application.ports import ContentSafetyClassifierPort


class NSFWClient(ContentSafetyClassifierPort):
    def classify(self, image_path):
        return (0.0, 1.0)