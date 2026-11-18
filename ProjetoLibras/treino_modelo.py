import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils import CSV_PATH, MODELO_PATH

MIN_AMOSTRAS = 30


def main(csv_path=CSV_PATH):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        sys.exit(f"[ERRO] Nenhum dado encontrado em '{csv_path}'. "
                 "Rode 'python coleta_dados.py' para coletar amostras primeiro.")

    df = pd.read_csv(csv_path, header=None)
    if df.empty:
        sys.exit(f"[ERRO] O arquivo '{csv_path}' está vazio. Colete amostras com 'python coleta_dados.py'.")

    y = df[0]
    X = df.drop(columns=0)

    contagem = y.value_counts().sort_index()
    print(f"{len(df)} amostras, {len(contagem)} letras.")
    for letra, n in contagem.items():
        if n < MIN_AMOSTRAS:
            print(f"[!] Letra '{letra}' tem só {n} amostras (recomendado: {MIN_AMOSTRAS}+).")

    if len(contagem) < 2:
        sys.exit("[ERRO] São necessárias pelo menos 2 letras diferentes para treinar.")

    # Divisão estratificada só se toda letra tiver amostras suficientes
    if contagem.min() >= 2 and len(df) * 0.2 >= len(contagem):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        print(f"\nAcurácia no teste: {accuracy_score(y_te, pred):.2%}\n")
        print(classification_report(y_te, pred, zero_division=0))
    else:
        print("[!] Amostras insuficientes para separar teste; treinando sem avaliação.")

    # Modelo final com todos os dados
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODELO_PATH)
    print(f"[OK] Modelo treinado e salvo em '{MODELO_PATH}'.")


if __name__ == "__main__":
    main()
