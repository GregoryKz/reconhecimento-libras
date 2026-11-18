import sys

import cv2
import joblib
import mediapipe as mp

from utils import MODELO_PATH, extrair_features

LIMIAR = 0.6

if not MODELO_PATH.exists():
    sys.exit(f"[✘] Modelo não encontrado em '{MODELO_PATH}'.\n"
             "Gere-o assim: 1) python coleta_dados.py  2) python treino_modelo.py")

modelo = joblib.load(MODELO_PATH)

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
            pontos = extrair_features(hand_landmarks)
            probs = modelo.predict_proba([pontos])[0]
            i = probs.argmax()
            confianca = probs[i]
            letra = modelo.classes_[i] if confianca >= LIMIAR else "?"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Letra: {letra} ({confianca:.0%})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Libras", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
