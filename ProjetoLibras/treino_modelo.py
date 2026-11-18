import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils import CSV_PATH, EXTERNO_PATH, MODELO_PATH

MIN_AMOSTRAS = 30

# Completa o treino com o dataset externo da UNIVALI (ver "Créditos" no README).
# Só entram as letras que NÃO existem em dados.csv, para que as amostras de
# outras pessoas não interfiram nas letras que você mesmo coletou.
USAR_EXTERNO = True

FONTE_PROPRIA = "própria"
FONTE_EXTERNA = "UNIVALI"


def _ler_csv(caminho):
    """Lê um CSV sem cabeçalho. Retorna None se o arquivo não existir ou estiver vazio."""
    if not caminho.exists() or caminho.stat().st_size == 0:
        return None
    df = pd.read_csv(caminho, header=None)
    return None if df.empty else df


def carregar_dados(csv_path=CSV_PATH, externo_path=EXTERNO_PATH, usar_externo=USAR_EXTERNO):
    """Junta as amostras próprias com as letras do dataset externo que não existem
    nas próprias. Retorna (df, fontes), onde fontes mapeia cada letra à sua origem
    (ou (None, {}) se não houver nenhuma amostra)."""
    partes = []
    fontes = {}

    proprio = _ler_csv(csv_path)
    if proprio is not None:
        partes.append(proprio)
        fontes.update(dict.fromkeys(proprio[0].unique(), FONTE_PROPRIA))

    if usar_externo:
        externo = _ler_csv(externo_path)
        if externo is None:
            print(f"[!] USAR_EXTERNO está ligado, mas '{externo_path}' não existe ou está vazio.")
        else:
            externo = externo[~externo[0].isin(set(fontes))]
            if not externo.empty:
                partes.append(externo)
                fontes.update(dict.fromkeys(externo[0].unique(), FONTE_EXTERNA))

    if not partes:
        return None, {}
    return pd.concat(partes, ignore_index=True), fontes


def main(csv_path=CSV_PATH, externo_path=EXTERNO_PATH):
    df, fontes = carregar_dados(csv_path, externo_path, USAR_EXTERNO)
    if df is None:
        sys.exit(f"[ERRO] Nenhum dado encontrado em '{csv_path}'. "
                 "Rode 'python coleta_dados.py' para coletar amostras primeiro.")

    y = df[0]
    X = df.drop(columns=0)

    contagem = y.value_counts().sort_index()
    print(f"{len(df)} amostras, {len(contagem)} letras.")
    for fonte in (FONTE_PROPRIA, FONTE_EXTERNA):
        letras = [letra for letra in contagem.index if fontes[letra] == fonte]
        if letras:
            print(f"  {fonte}: {contagem[letras].sum()} amostras ({', '.join(letras)})")

    print(f"\n{'Letra':<6} {'Fonte':<9} {'Amostras':>8}")
    for letra, n in contagem.items():
        aviso = f"  [!] menos de {MIN_AMOSTRAS} (recomendado: {MIN_AMOSTRAS}+)" if n < MIN_AMOSTRAS else ""
        print(f"{letra:<6} {fontes[letra]:<9} {n:>8}{aviso}")

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
        letras = list(clf.classes_)
        nomes = [f"{letra} ({fontes[letra]})" for letra in letras]
        print(classification_report(y_te, pred, labels=letras, target_names=nomes, zero_division=0))
    else:
        print("[!] Amostras insuficientes para separar teste; treinando sem avaliação.")

    # Modelo final com todos os dados
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODELO_PATH)
    print(f"[OK] Modelo treinado e salvo em '{MODELO_PATH}'.")


if __name__ == "__main__":
    main()
