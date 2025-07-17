
#pip install opencv-python mediapipe numpy pandas scikit-learn joblib

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Cria pasta para salvar os dados se não existir
if not os.path.exists("dataset_libras"):
    os.makedirs("dataset_libras")

# Inicializa MediaPipe para capturar mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Captura de vídeo
cap = cv2.VideoCapture(0)

print("Pressione uma letra (A-Z) para rotular o gesto.")
print("Depois pressione 'S' para salvar os pontos da mão com esse rótulo.")
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

    cv2.imshow("Coleta de Dados - Libras", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key != -1:
        char = chr(key).upper()
        if char == 'S' and resultado.multi_hand_landmarks:
            # Salvar os pontos
            for hand_landmarks in resultado.multi_hand_landmarks:
                pontos = []
                for lm in hand_landmarks.landmark:
                    pontos.extend([lm.x, lm.y, lm.z])

                if letra_atual:
                    with open('dataset_libras/dados.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([letra_atual] + pontos)
                        print(f"[✔] Letra '{letra_atual}' salva.")
        elif char.isalpha():
            letra_atual = char
            print(f"Letra atual definida: {letra_atual}")

cap.release()
cv2.destroyAllWindows()
