def accept_sample(features, label, phys_risk):
    # Ziel:
    # schlechte / widersprüchliche Daten rauswerfen

    # Wenn Label SAFE aber Physik sagt hoch gefährlich → verwerfen
    if label == 0 and phys_risk > 0.7:
        return False

    # Wenn Label DANGER aber keine Physik → schwaches Sample
    if label == 1 and phys_risk < 0.2:
        return False

    return True
