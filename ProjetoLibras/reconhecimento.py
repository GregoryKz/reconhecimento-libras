import cv2
import mediapipe as mp
import numpy as np
import joblib

# Carrega o modelo
modelo = joblib.load("modelo_letras_libras.joblib")

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            pontos = []
            for lm in hand_landmarks.landmark:
                pontos.extend([lm.x, lm.y, lm.z])

            # Faz a predição com o modelo
            predicao = modelo.predict([pontos])
            letra = predicao[0]

            # Desenha a mão e a letra reconhecida
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Letra: {letra}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Libras", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
