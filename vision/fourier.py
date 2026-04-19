import numpy as np

def flicker_frequency(window, fps=30):
    """
    Erkennt dominante Flimmerfrequenz in Hz
    """

    arr = np.array(window).astype(np.float32)

    # 1. Zeitreihe (Helligkeit pro Frame)
    signal = np.mean(arr, axis=(1, 2))

    # 2. Fourier Transform
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fps)

    power = np.abs(fft)

    # nur positive Frequenzen
    mask = freqs > 0
    freqs = freqs[mask]
    power = power[mask]

    if len(power) == 0:
        return 0.0

    dominant_freq = freqs[np.argmax(power)]

    return float(dominant_freq)
