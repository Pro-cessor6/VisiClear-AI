import cv2
import numpy as np
import mss
import time

from vision.features import extract_features
from vision.physics import physics_risk
from model.net import RiskNet
from vision.safety_engine import SafetyEngine

# =========================
# INIT
# =========================
model = RiskNet()
model.eval()

safety_engine = SafetyEngine()

WINDOW_SIZE = 30
buffer = []

smoothed_risk = 0.0
overlay_active = True

print("🛡️ ClearVision Overlay gestartet (F6 = toggle, ESC = exit)")

# =========================
# FLICKER MAP (FIXED)
# =========================
def flicker_map(buffer):
    arr = np.array(buffer, dtype=np.float32) / 255.0

    h, w = arr.shape[1], arr.shape[2]
    block = 32

    mask = np.zeros((h, w), dtype=np.float32)

    # 🔥 real temporal diff (FIXED)
    diff_stack = np.abs(np.diff(arr, axis=0))

    for y in range(0, h, block):
        for x in range(0, w, block):

            region = diff_stack[:, y:y+block, x:x+block]

            if region.size == 0:
                continue

            mask[y:y+block, x:x+block] = np.mean(region)

    # safe normalization
    mask = mask / (np.max(mask) + 1e-6)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return mask[..., None]

# =========================
# SCREEN CAPTURE
# =========================
with mss.mss() as sct:

    monitor = sct.monitors[1]

    cv2.namedWindow("ClearVision", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ClearVision", cv2.WND_PROP_TOPMOST, 1)

    while True:

        # =========================
        # RAW SCREEN
        # =========================
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (640, 360))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        buffer.append(gray)
        if len(buffer) > WINDOW_SIZE:
            buffer.pop(0)

        display = frame.copy()

        # =========================
        # RISK (FIXED FUSION)
        # =========================
        if len(buffer) == WINDOW_SIZE:

            features = extract_features(buffer)
            phys = float(physics_risk(features))

            try:
                import torch
                with torch.no_grad():
                    inp = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    model_risk = float(model(inp).item())
            except:
                model_risk = 0.0

            # 🔥 stabilized fusion (FIXED)
            phys = np.clip(phys, 0.0, 1.0)
            model_risk = np.clip(model_risk, 0.0, 1.0)

            risk = np.sqrt(0.5 * model_risk**2 + 0.5 * phys**2)

            smoothed_risk = 0.8 * smoothed_risk + 0.2 * risk
            smoothed_risk = np.clip(smoothed_risk, 0.0, 1.0)

        else:
            smoothed_risk = 0.0

        # =========================
        # OVERLAY (FIXED PIPELINE)
        # =========================
        if overlay_active and len(buffer) == WINDOW_SIZE:

            mask = flicker_map(buffer)
            mask = np.clip(mask, 0.0, 1.0)

            alpha = mask * smoothed_risk * 0.8
            alpha = np.clip(alpha, 0.0, 1.0)

            # float-safe blending
            display_f = display.astype(np.float32) / 255.0
            display_f *= (1.0 - alpha)

            display = (display_f * 255.0).astype(np.uint8)

            display = safety_engine.process(display, smoothed_risk)

            cv2.putText(
                display,
                f"SAFE MODE | Risk {smoothed_risk:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        else:
            cv2.putText(
                display,
                "RAW MODE (F6)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # =========================
        # SHOW
        # =========================
        cv2.imshow("ClearVision", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 117:  # F6
            overlay_active = not overlay_active

        if key == 27:  # ESC
            break

cv2.destroyAllWindows()
