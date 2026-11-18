import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

from utils import MODELO_PATH, centro_palma, extrair_features, gesto_confirmar, gesto_mao_aberta

LIMIAR = 0.6                # confiança mínima para aceitar a letra prevista
TEMPO_CANDIDATA = 0.5       # s que a mesma letra precisa se manter para virar candidata
TEMPO_CONFIRMAR = 0.4       # s segurando o punho para baixo para confirmar a candidata
TEMPO_ESPACO = 2.0          # s com a mão aberta parada para inserir um espaço
TOLERANCIA_MOVIMENTO = 0.3  # quanto a palma pode se mover (em tamanhos de palma) e ainda contar como parada

ALTURA_FAIXA = 100
FONTE = cv2.FONT_HERSHEY_SIMPLEX
VERDE, AMARELO, BRANCO, CINZA = (0, 255, 0), (0, 255, 255), (255, 255, 255), (160, 160, 160)
TECLA_ESC = 27
TECLAS_BACKSPACE = (8, 127)  # 127: backspace no macOS


class Temporizador:
    """Dispara uma única vez quando o mesmo valor (letra ou True) se mantém por
    `duracao` segundos. Mudar o valor, ou passar None/False, recomeça a contagem."""

    def __init__(self, duracao):
        self.duracao = duracao
        self.valor = None
        self.inicio = 0.0
        self.disparou = False

    def atualizar(self, valor, agora):
        valor = valor or None
        if valor != self.valor:
            self.valor, self.inicio, self.disparou = valor, agora, False
        if valor is None or self.disparou or agora - self.inicio < self.duracao:
            return False
        self.disparou = True
        return True

    def reiniciar(self, agora):
        self.inicio = agora

    def progresso(self, agora):
        if self.valor is None:
            return 0.0
        if self.disparou:
            return 1.0
        return min((agora - self.inicio) / self.duracao, 1.0)


def ultimo_trecho(texto, largura_max, escala, espessura):
    """Se o texto não couber na largura, mostra só o final dele."""
    if cv2.getTextSize(texto, FONTE, escala, espessura)[0][0] <= largura_max:
        return texto
    while texto and cv2.getTextSize("..." + texto, FONTE, escala, espessura)[0][0] > largura_max:
        texto = texto[1:]
    return "..." + texto


def desenhar_faixa(frame, gesto, progresso, candidata, texto):
    h, w = frame.shape[:2]
    y0 = h - ALTURA_FAIXA

    rotulo = f"Gesto: {gesto or '-'}"
    cv2.putText(frame, rotulo, (10, y0 + 28), FONTE, 0.7, AMARELO if gesto else CINZA, 2)
    if gesto:
        x = 20 + cv2.getTextSize(rotulo, FONTE, 0.7, 2)[0][0]
        cv2.rectangle(frame, (x, y0 + 14), (x + 120, y0 + 28), CINZA, 1)
        cv2.rectangle(frame, (x, y0 + 14), (x + int(120 * progresso), y0 + 28),
                      VERDE if progresso >= 1 else AMARELO, -1)

    cv2.putText(frame, f"Candidata: {candidata or '-'}", (10, y0 + 58), FONTE, 0.7, BRANCO, 2)

    prefixo = "Texto: "
    largura_prefixo = cv2.getTextSize(prefixo, FONTE, 0.9, 2)[0][0]
    visivel = ultimo_trecho(texto + "_", w - 20 - largura_prefixo, 0.9, 2)
    cv2.putText(frame, prefixo + visivel, (10, y0 + 88), FONTE, 0.9, VERDE, 2)


if not MODELO_PATH.exists():
    sys.exit(f"[ERRO] Modelo não encontrado em '{MODELO_PATH}'.\n"
             "Gere-o assim: 1) python coleta_dados.py  2) python treino_modelo.py")

modelo = joblib.load(MODELO_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

texto = ""
candidata = None
estavel = Temporizador(TEMPO_CANDIDATA)
confirmar = Temporizador(TEMPO_CONFIRMAR)
espaco = Temporizador(TEMPO_ESPACO)
ancora = None  # centro da palma de onde a mão aberta começou a contar como parada

while True:
    ret, frame = cap.read()
    if not ret:
        break
    agora = time.monotonic()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)
    mao = resultado.multi_hand_landmarks[0] if resultado.multi_hand_landmarks else None

    eh_confirmar = mao is not None and gesto_confirmar(mao)
    eh_aberta = mao is not None and not eh_confirmar and gesto_mao_aberta(mao)

    # Escurece a faixa de baixo antes de desenhar a mão, para os pontos ficarem visíveis
    frame[-ALTURA_FAIXA:] //= 3

    letra = None
    if mao is not None:
        mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)
        if not eh_confirmar:  # o gesto CONFIRMAR não passa pelo modelo
            probs = modelo.predict_proba([extrair_features(mao)])[0]
            i = probs.argmax()
            if probs[i] >= LIMIAR:
                letra = str(modelo.classes_[i])
            cv2.putText(frame, f"Letra: {letra or '?'} ({probs[i]:.0%})", (10, 50),
                        FONTE, 1, VERDE, 2)

    # Candidata: última letra prevista com confiança que ficou estável
    if estavel.atualizar(letra, agora):
        candidata = letra

    # CONFIRMAR: adiciona a candidata ao texto e a limpa
    if confirmar.atualizar(eh_confirmar, agora) and candidata:
        texto += candidata
        candidata = None

    # ESPAÇO: mão aberta parada; se a palma se mover demais, a contagem recomeça
    if eh_aberta:
        centro, tamanho = centro_palma(mao)
        if ancora is None or np.linalg.norm(centro - ancora) > TOLERANCIA_MOVIMENTO * tamanho:
            ancora = centro
            espaco.reiniciar(agora)
    else:
        ancora = None
    if espaco.atualizar(eh_aberta, agora):
        if texto and not texto.endswith(" "):
            texto += " "
        candidata = None  # descarta letra que o modelo tenha visto na mão aberta

    if eh_confirmar:
        gesto, progresso = "CONFIRMAR", confirmar.progresso(agora)
    elif eh_aberta:
        gesto, progresso = "ESPACO", espaco.progresso(agora)
    else:
        gesto, progresso = None, 0.0

    cv2.putText(frame, "BACKSPACE = apagar | C = limpar | ESC = sair", (10, 80),
                FONTE, 0.5, BRANCO, 1)
    desenhar_faixa(frame, gesto, progresso, candidata, texto)
    cv2.imshow("Reconhecimento de Libras", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == TECLA_ESC:
        break
    elif tecla in TECLAS_BACKSPACE:
        texto = texto[:-1]
    elif tecla in (ord("c"), ord("C")):
        texto = ""
        candidata = None

cap.release()
cv2.destroyAllWindows()

if texto.strip():
    print(f"Texto montado: {texto}")
