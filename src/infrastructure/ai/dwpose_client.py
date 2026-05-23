from src.application.ports import PoseExtractionPort


class DWPoseClient(PoseExtractionPort):
    def extract_keypoints(self, image_path):
        return {"body": [], "face": [], "left_hand": [], "right_hand": []}