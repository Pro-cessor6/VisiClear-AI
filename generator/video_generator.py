import cv2
import numpy as np
import random
import os


# =========================
# MODE SELECTION
# =========================
def choose_mode():
    r = random.random()

    if r < 0.4:
        return "SAFE"
    elif r < 0.8:
        return "DANGER"
    else:
        return "MIXED"


# =========================
# VIDEO GENERATION
# =========================
def generate_video(path, mode, colors, freq, duration=3, fps=30):

    w, h = 640, 480
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frames_per_cycle = max(1, int(fps / max(freq, 0.1)))

    for i in range(fps * duration):

        # ================= SAFE =================
        if mode == "SAFE":
            frame = np.full((h, w, 3), random.choice(colors), dtype=np.uint8)

            noise = np.random.normal(0, 3, frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        # ================= DANGER =================
        elif mode == "DANGER":
            idx = (i // frames_per_cycle) % len(colors)
            frame = np.full((h, w, 3), colors[idx], dtype=np.uint8)

            if (i // frames_per_cycle) % 2 == 0:
                frame[:] = (0, 0, 0)
            else:
                frame[:] = (255, 255, 255)

            noise = np.random.normal(0, 12, frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        # ================= MIXED =================
        else:
            mid = w // 2
            frame = np.full((h, w, 3), random.choice(colors), dtype=np.uint8)

            if (i // frames_per_cycle) % 2 == 0:
                frame[:, mid:] = (0, 0, 0)

            noise = np.random.normal(0, 8, frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        out.write(frame)

    out.release()
    print(f"✅ Video gespeichert: {path} ({mode})")


# =========================
# PRE-GENERATE BATCH
# =========================
def pre_generate_batch(colors, freq):

    os.makedirs("data/videos", exist_ok=True)

    batch = []

    for i in range(2):  # 2 Videos voraus
        mode = choose_mode()
        label = 0 if mode == "SAFE" else 1

        path = f"data/videos/video_{i}_{mode}.mp4"

        generate_video(path, mode, colors, freq)

        batch.append((path, label))

    return batch
