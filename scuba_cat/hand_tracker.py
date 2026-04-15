import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

class HandTracker:
    def __init__(self, max_hands=2, det_conf=0.6, trk_conf=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        hands_np = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
                hands_np.append(pts)
        return results, hands_np

    def draw(self, frame_bgr, results):
        if not results.multi_hand_landmarks:
            return frame_bgr
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        return frame_bgr

    def close(self):
        self.hands.close()
