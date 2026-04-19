import cv2
import numpy as np

class SafetyEngine:
    def __init__(self):
        self.prev_frame = None

    def process(self, frame, risk):

        frame = frame.astype(np.float32)

        # =========================
        # TEMPORAL SMOOTHING
        # =========================
        if self.prev_frame is None:
            self.prev_frame = frame

        alpha = 0.6 if risk < 0.5 else 0.3
        frame = alpha * frame + (1 - alpha) * self.prev_frame

        # =========================
        # CONTRAST CONTROL
        # =========================
        if risk > 0.4:
            frame = frame * 0.9 + 20

        # =========================
        # FLICKER DAMPING
        # =========================
        if risk > 0.6:
            frame = cv2.GaussianBlur(frame, (9, 9), 0)

        self.prev_frame = frame.copy()

        return np.clip(frame, 0, 255).astype(np.uint8)
