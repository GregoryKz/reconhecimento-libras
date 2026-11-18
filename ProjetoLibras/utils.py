from pathlib import Path

import numpy as np

PASTA = Path(__file__).parent
CSV_PATH = PASTA / "dataset_libras" / "dados.csv"
MODELO_PATH = PASTA / "modelo_letras_libras.joblib"


def extrair_features(hand_landmarks):
    """Normaliza os 21 pontos da mão: subtrai o pulso (landmark 0) e divide
    pela maior distância ao pulso. Retorna lista de 63 valores (x, y, z)."""
    pontos = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    pontos = pontos - pontos[0]
    maior = np.linalg.norm(pontos, axis=1).max()
    if maior > 0:
        pontos = pontos / maior
    return pontos.flatten().tolist()
