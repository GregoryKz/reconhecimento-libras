from pathlib import Path

import numpy as np

PASTA = Path(__file__).parent
CSV_PATH = PASTA / "dataset_libras" / "dados.csv"
EXTERNO_PATH = PASTA / "dataset_libras" / "externo_univali.csv"
MODELO_PATH = PASTA / "modelo_letras_libras.joblib"

# Índices dos landmarks do MediaPipe Hands
PULSO = 0
PONTAS = (8, 12, 16, 20)  # indicador, médio, anelar, mínimo
MCPS = (5, 9, 13, 17)     # articulação da base de cada um desses dedos
# Cadeias de pontos de cada dedo, do pulso (ou base do polegar) até a ponta
CADEIAS_DEDOS = ((0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
CADEIA_POLEGAR = (1, 2, 3, 4)

# Um dedo conta como esticado se a distância direta da base até a ponta for pelo
# menos esta fração do comprimento somado dos seus segmentos (1.0 = reto).
RETIDAO_MIN = 0.9


def extrair_features(hand_landmarks):
    """Normaliza os 21 pontos da mão: subtrai o pulso (landmark 0) e divide
    pela maior distância ao pulso. Retorna lista de 63 valores (x, y, z)."""
    pontos = _pontos(hand_landmarks)
    pontos = pontos - pontos[0]
    maior = np.linalg.norm(pontos, axis=1).max()
    if maior > 0:
        pontos = pontos / maior
    return pontos.flatten().tolist()


def _pontos(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _esticado(pontos, cadeia):
    comprimento = sum(_dist(pontos[a], pontos[b]) for a, b in zip(cadeia, cadeia[1:]))
    if comprimento == 0:
        return False
    return _dist(pontos[cadeia[0]], pontos[cadeia[-1]]) / comprimento >= RETIDAO_MIN


def gesto_confirmar(hand_landmarks):
    """Punho fechado apontando para baixo: as pontas dos 4 dedos estão abaixo do
    pulso no eixo y da imagem (y cresce para baixo) e cada dedo está dobrado
    (ponta mais perto do pulso do que a MCP do mesmo dedo)."""
    p = _pontos(hand_landmarks)
    pulso = p[PULSO]
    for ponta, mcp in zip(PONTAS, MCPS):
        if p[ponta][1] <= pulso[1]:
            return False
        if _dist(p[ponta], pulso) >= _dist(p[mcp], pulso):
            return False
    return True


def gesto_mao_aberta(hand_landmarks):
    """Mão aberta com os dedos para cima: os 5 dedos esticados e as pontas dos 4
    dedos acima do pulso (exigir "para cima" evita confundir com letras de dedos
    esticados para baixo, como o M). O polegar também precisa estar afastado da
    palma (ponta mais longe da base do mínimo do que a articulação IP), para não
    confundir com sinais em que ele fica dobrado sobre a palma, como o B."""
    p = _pontos(hand_landmarks)
    if any(p[ponta][1] >= p[PULSO][1] for ponta in PONTAS):
        return False
    if not all(_esticado(p, cadeia) for cadeia in CADEIAS_DEDOS):
        return False
    return _esticado(p, CADEIA_POLEGAR) and _dist(p[4], p[17]) > _dist(p[3], p[17])


def centro_palma(hand_landmarks):
    """Retorna (centro, tamanho) da palma no plano da imagem: o centro é a média
    do pulso e das 4 MCPs, e o tamanho é a distância do pulso à MCP do médio.
    Serve para medir se a mão está parada independentemente da distância à câmera."""
    p = _pontos(hand_landmarks)[:, :2]
    centro = p[[PULSO, *MCPS]].mean(axis=0)
    return centro, _dist(p[PULSO], p[9])
