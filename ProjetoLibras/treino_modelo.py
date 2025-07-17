from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# Carrega os dados do CSV
df = pd.read_csv("dataset_libras/dados.csv", header=None)

# Separa as colunas
y = df[0]               # A primeira coluna é a letra
X = df.drop(columns=0)  # O restante são os pontos (x, y, z) da mão

# Cria e treina o modelo
clf = RandomForestClassifier()
clf.fit(X, y)

# Salva o modelo em disco
joblib.dump(clf, "modelo_letras_libras.joblib")

print("[✔] Modelo treinado e salvo como 'modelo_letras_libras.joblib'.")
