# Reconhecimento de Libras

Reconhece letras do alfabeto manual de Libras (Língua Brasileira de Sinais) em tempo real pela webcam. O [MediaPipe Hands](https://developers.google.com/mediapipe) extrai 21 pontos da mão, que são normalizados e classificados por um `RandomForest` (scikit-learn) treinado com os seus próprios dados.

## Estrutura

```
ProjetoLibras/
├── coleta_dados.py      # 1. coleta amostras pela webcam
├── treino_modelo.py     # 2. treina o modelo
├── reconhecimento.py    # 3. reconhece em tempo real e monta palavras
├── utils.py             # normalização dos pontos, gestos e caminhos (usado pelos 3)
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

### 3. Reconhecer e montar palavras

```bash
python ProjetoLibras/reconhecimento.py
```

No topo aparece a letra prevista e a confiança (`?` se a confiança for menor que 60%). A faixa de baixo mostra o gesto detectado (com uma barra de progresso), a letra candidata e o texto montado.

Para escrever:

1. Faça a letra e segure. Quando a mesma letra é prevista com confiança de 60% ou mais por ~0,5 s, ela vira a **candidata**, e continua candidata mesmo depois que você muda a mão.
2. Faça o gesto **CONFIRMAR** e segure por ~0,4 s: a candidata entra no texto e é limpa. Para repetir uma letra, faça-a de novo e confirme de novo.
3. Para separar palavras, faça o gesto **ESPAÇO** e segure a mão parada por ~2 s. Não entra espaço duplicado nem no começo do texto. O espaço também descarta a candidata, para não sobrar uma letra que o modelo tenha visto na mão aberta.

| Gesto | Como fazer | Tempo |
|---|---|---|
| CONFIRMAR | Punho fechado apontando para baixo: pontas dos 4 dedos abaixo do pulso na imagem e dedos dobrados. | ~0,4 s |
| ESPAÇO | Mão aberta com os 5 dedos esticados e apontando para cima, parada. Se a mão se mexer, a contagem recomeça. | ~2 s |

Os gestos são reconhecidos por regras sobre os pontos da mão (`gesto_confirmar` e `gesto_mao_aberta` em `utils.py`), não pelo modelo. Enquanto o CONFIRMAR está ativo, o modelo não é consultado, para o gesto não virar uma letra.

Teclas:

- **BACKSPACE**: apaga a última letra.
- **C**: limpa o texto e a candidata.
- **ESC**: sai e imprime o texto montado no terminal.

Os tempos e o limiar são constantes no topo do `reconhecimento.py`: `LIMIAR`, `TEMPO_CANDIDATA`, `TEMPO_CONFIRMAR`, `TEMPO_ESPACO` e `TOLERANCIA_MOVIMENTO` (quanto a palma pode se mover, em tamanhos de palma, sem recomeçar a contagem do ESPAÇO).

## Limitações

- Funciona com **um único frame**, então letras que dependem de **movimento** (J, H, K, X, Z e Ç, conforme o alfabeto manual usado) não são reconhecidas de forma confiável.
- Só detecta uma mão por vez.
- Os gestos CONFIRMAR e ESPAÇO dependem da orientação da mão na imagem (para baixo e para cima). Uma letra que seja um punho fechado para baixo nunca será prevista, e uma letra com os 5 dedos esticados para cima pode inserir um espaço se for segurada por 2 s.
- O desempenho depende da qualidade e da variedade dos dados que você coletar; o modelo é treinado por pessoa/ambiente.
- Os dados coletados com versões anteriores (pontos sem normalização) são incompatíveis e devem ser recoletados.

## Autor

Gregory Klaus — projeto de estudo em visão computacional e acessibilidade.
