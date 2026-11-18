# Reconhecimento de Libras

Reconhece letras do alfabeto manual de Libras (Língua Brasileira de Sinais) em tempo real pela webcam. O [MediaPipe Hands](https://developers.google.com/mediapipe) extrai 21 pontos da mão, que são normalizados e classificados por um `RandomForest` (scikit-learn) treinado com os seus próprios dados.

## Estrutura

```
ProjetoLibras/
├── coleta_dados.py      # 1. coleta amostras pela webcam
├── treino_modelo.py     # 2. treina o modelo
├── reconhecimento.py    # 3. reconhece em tempo real
├── utils.py             # normalização dos pontos e caminhos (usado pelos 3)
└── dataset_libras/
    └── dados.csv        # amostras coletadas (rótulo + 63 valores normalizados)
requirements.txt
```

O modelo gerado (`ProjetoLibras/modelo_letras_libras.joblib`) não é versionado.

## Requisitos

- **Python 3.12 (obrigatório).** O código usa `mp.solutions` (API legada do MediaPipe), que só existe até o `mediapipe==0.10.21`. Esse pacote não tem versão para Python 3.13 no Windows, e as versões novas do MediaPipe removeram `solutions`.
- Webcam.

## Criando o ambiente

```bash
conda create -n libras python=3.12 -y
conda activate libras
pip install -r requirements.txt
python -c "import mediapipe as mp; mp.solutions.hands.Hands(); print('ok')"
```

> Se o conda reclamar dos Termos de Serviço dos canais, use `-c conda-forge --override-channels` no `conda create`.

## Uso

Os scripts funcionam a partir de qualquer diretório.

### 1. Coletar dados

```bash
python ProjetoLibras/coleta_dados.py
```

- Tecla **A–Z**: escolhe a letra que você vai gravar.
- **ESPAÇO**: salva a amostra da mão atual com a letra escolhida.
- **ESC**: sai.

A tela mostra a letra atual e quantas amostras ela já tem. Colete **30 ou mais amostras por letra**, variando um pouco posição, distância e ângulo da mão.

### 2. Treinar

```bash
python ProjetoLibras/treino_modelo.py
```

Imprime a acurácia e o `classification_report` num conjunto de teste e avisa sobre letras com menos de 30 amostras. Gera `ProjetoLibras/modelo_letras_libras.joblib`.

### 3. Reconhecer

```bash
python ProjetoLibras/reconhecimento.py
```

Mostra a letra prevista e a confiança. Se a confiança for menor que 60%, mostra `?`. ESC sai.

## Limitações

- Funciona com **um único frame**, então letras que dependem de **movimento** (J, H, K, X, Z e Ç, conforme o alfabeto manual usado) não são reconhecidas de forma confiável.
- Só detecta uma mão por vez.
- O desempenho depende da qualidade e da variedade dos dados que você coletar; o modelo é treinado por pessoa/ambiente.
- Os dados coletados com versões anteriores (pontos sem normalização) são incompatíveis e devem ser recoletados.

## Autor

Gregory Klaus — projeto de estudo em visão computacional e acessibilidade.
