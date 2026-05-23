from src.application.ports import ImageSegmentationPort


class SAM3Client(ImageSegmentationPort):
    def crop_image(self, image_path, mode="square"):
        return image_path