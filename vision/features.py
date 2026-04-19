import numpy as np

def extract_features(buffer, fps=30):
    arr = np.array(buffer).astype(np.float32)

    # =========================
    # 1. GLOBAL SIGNAL
    # =========================
    signal = np.mean(arr, axis=(1, 2))

    # =========================
    # 2. BRIGHTNESS
    # =========================
    brightness = np.mean(signal)

    # =========================
    # 3. FLICKER ENERGY (stabil!)
    # =========================
    diff = np.diff(signal)
    flicker_energy = np.std(diff)

    # =========================
    # 4. FREQUENCY (Fourier)
    # =========================
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)

    magnitudes = np.abs(fft)

    # DC (0 Hz) ignorieren
    magnitudes[0] = 0

    dominant_freq = freqs[np.argmax(magnitudes)]

    # =========================
    # NORMALISIERUNG
    # =========================
    brightness /= 255.0
    flicker_energy /= 50.0   # grob skaliert
    dominant_freq /= 30.0    # max ~30Hz

    return np.array([brightness, flicker_energy, dominant_freq], dtype=np.float32)
