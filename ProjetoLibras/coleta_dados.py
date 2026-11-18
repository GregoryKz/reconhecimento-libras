import csv
from collections import Counter

import cv2
import mediapipe as mp

from utils import CSV_PATH, extrair_features

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

# Conta as amostras já existentes por letra
contagem = Counter()
if CSV_PATH.exists():
    with open(CSV_PATH, newline="") as f:
        for linha in csv.reader(f):
            if linha:
                contagem[linha[0]] += 1

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Pressione uma letra (A-Z) para escolher o gesto.")
print("Pressione ESPAÇO para salvar a amostra da mão com essa letra.")
print("Pressione ESC para sair.")

letra_atual = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if letra_atual:
        texto = f"Letra: {letra_atual} | Amostras: {contagem[letra_atual]}"
    else:
        texto = "Escolha uma letra (A-Z)"
    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "ESPACO = salvar | ESC = sair", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Coleta de Dados - Libras", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # ESPAÇO
        if letra_atual and resultado.multi_hand_landmarks:
            pontos = extrair_features(resultado.multi_hand_landmarks[0])
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([letra_atual] + pontos)
            contagem[letra_atual] += 1
            print(f"[✔] Letra '{letra_atual}' salva ({contagem[letra_atual]} amostras).")
        elif not letra_atual:
            print("Escolha uma letra antes de salvar.")
        else:
            print("Nenhuma mão detectada.")
    elif key != 255 and chr(key).isalpha() and chr(key).isascii():
        letra_atual = chr(key).upper()
        print(f"Letra atual definida: {letra_atual}")

cap.release()
cv2.destroyAllWindows()
