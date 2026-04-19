import numpy as np

def physics_risk(features):
    brightness, energy, freq = features

    risk = 0.0

    # kritischer Bereich (Epilepsie relevant)
    if 3 <= freq <= 30:
        risk += 0.6

    # starke Flicker Energie
    if energy > 20:
        risk += 0.3

    # extrem hohe Kontraste (optional)
    if brightness < 30 or brightness > 220:
        risk += 0.1

    return min(risk, 1.0)
