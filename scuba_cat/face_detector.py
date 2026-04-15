import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection


class FaceDetector:
    NOSE_TIP_INDEX = 2

    def __init__(self, min_confidence: float = 0.5):
        self.face = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence,
        )

    def nose_tip(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face.process(rgb)
        if not results.detections:
            return None
        det = max(results.detections, key=lambda d: d.score[0])
        kp = det.location_data.relative_keypoints[self.NOSE_TIP_INDEX]
        return (float(kp.x), float(kp.y))

    def close(self):
        self.face.close()
