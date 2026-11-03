"""
=====================================================================================
 Previsão Direcional (Alta/Baixa) com Séries Temporais + Análise de Sentimento
 Arquitetura híbrida CNN-LSTM com dois ramos (preço | sentimento) - TensorFlow/Keras
=====================================================================================
 Ferramentas 100% gratuitas:
   pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
   (opcional, para gerar sentimento real) pip install transformers torch

 Pode ser executado no Google Colab gratuito (CPU ou GPU T4).

 Fluxo:
   1. Aquisição (yfinance + CSV de sentimento ou dados sintéticos)
   2. Alinhamento por data (notícias de fim de semana/feriado -> próximo pregão)
   3. Engenharia de features + Target (fechamento de t+1 > fechamento de t)
   4. Divisão cronológica com "purga" na fronteira (anti data leakage)
   5. MinMaxScaler ajustado SOMENTE no treino
   6. Janelas de 60 dias
   7. Modelo CNN-LSTM com Dropout + Adam
   8. Avaliação: Classification Report, Matriz de Confusão, Acurácia Direcional,
      baseline ingênuo e mini-backtest
=====================================================================================
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score, roc_auc_score)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------
# 0. CONFIGURAÇÃO
# -----------------------------------------------------------------------------------
CONFIG = {
    "ticker": "^BVSP",               # Ibovespa. Ex. alternativos: "PETR4.SA", "VALE3.SA"
    "inicio": "2015-01-01",
    "fim": None,                      # None = até hoje
    "csv_sentimento": "sentimento.csv",  # colunas obrigatórias: Data, Sentimento_Score
    "janela": 60,                     # dias de observação por amostra
    "prop_treino": 0.80,              # 80% treino / 20% teste (cronológico)
    "prop_validacao": 0.15,           # fatia FINAL do treino usada para early stopping
    "epocas": 100,
    "batch": 32,
    "seed": 42,
    "pasta_saida": "artefatos",       # onde dataset e modelo serão salvos
}


def fixar_sementes(seed: int) -> None:
    """Reprodutibilidade (na medida do possível em GPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------------------------------------------------------------
# 1. AQUISIÇÃO DE DADOS
# -----------------------------------------------------------------------------------
def baixar_precos(ticker: str, inicio: str, fim=None) -> pd.DataFrame:
    """Baixa OHLCV diário pelo yfinance (gratuito)."""
    df = yf.download(ticker, start=inicio, end=fim, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {ticker}. Verifique o ticker/conexão.")

    # Versões recentes do yfinance retornam colunas MultiIndex (Campo, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Data"
    return df.dropna()


def gerar_sentimento_sintetico(precos: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Gera um score de sentimento em [-1, 1] alinhado às datas do yfinance, apenas para
    testar o pipeline. Ele é levemente correlacionado com o retorno do MESMO dia
    (notícia reage ao mercado) e NÃO contém informação do futuro, então o modelo
    não deve "trapacear". Espere acurácia próxima de 50% com ele.
    """
    rng = np.random.default_rng(seed)
    ret = np.log(precos["Adj Close"]).diff().fillna(0.0)
    z = ret / (ret.std() + 1e-12)
    score = np.tanh(0.5 * z + rng.normal(0.0, 1.0, len(z)))
    return pd.DataFrame({
        "Data": precos.index.to_numpy(),
        "Sentimento_Score": np.clip(np.asarray(score), -1, 1),
    })


def carregar_sentimento(caminho: str, precos: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Carrega o CSV externo; se não existir, cai para dados sintéticos."""
    if os.path.exists(caminho):
        print(f"[INFO] Carregando sentimento de '{caminho}'")
        s = pd.read_csv(caminho)
        faltando = {"Data", "Sentimento_Score"} - set(s.columns)
        if faltando:
            raise ValueError(f"CSV de sentimento sem as colunas: {faltando}")
        s["Data"] = pd.to_datetime(s["Data"]).dt.tz_localize(None).dt.normalize()
        s["Sentimento_Score"] = pd.to_numeric(s["Sentimento_Score"], errors="coerce").clip(-1, 1)
        return s[["Data", "Sentimento_Score"]].dropna()

    print(f"[AVISO] '{caminho}' não encontrado -> usando sentimento SINTÉTICO.")
    return gerar_sentimento_sintetico(precos, seed)


def gerar_sentimento_finbert(csv_noticias: str, coluna_texto: str = "Titulo",
                             saida: str = "sentimento.csv",
                             modelo: str = "ProsusAI/finbert") -> None:
    """
    OPCIONAL - transforma manchetes em score de sentimento com FinBERT (Hugging Face,
    gratuito). Entrada: CSV com colunas 'Data' e a coluna de texto.
    Score = P(positivo) - P(negativo), no intervalo [-1, 1].

    - 'ProsusAI/finbert' é treinado em inglês (ex.: datasets do Kaggle como
      "Daily Financial News for 6000+ Stocks").
    - Para manchetes em português, procure no Hugging Face um FinBERT-PT-BR e
      confira os nomes dos rótulos do modelo escolhido.
    """
    from transformers import pipeline  # import tardio: só exige a lib se usar

    clf = pipeline("text-classification", model=modelo, top_k=None, truncation=True)
    df = pd.read_csv(csv_noticias)
    resultados = clf(df[coluna_texto].astype(str).tolist(), batch_size=32)

    def score(r):
        p = {d["label"].lower(): d["score"] for d in r}
        return p.get("positive", 0.0) - p.get("negative", 0.0)

    df["Sentimento_Score"] = [score(r) for r in resultados]
    df[["Data", "Sentimento_Score"]].to_csv(saida, index=False)
    print(f"[INFO] Sentimento FinBERT salvo em '{saida}' ({len(df)} manchetes).")


# -----------------------------------------------------------------------------------
# 2. ALINHAMENTO (MERGE) POR DATA
# -----------------------------------------------------------------------------------
def alinhar_dados(precos: pd.DataFrame, sentimento: pd.DataFrame) -> pd.DataFrame:
    """
    - Notícias de fim de semana/feriado são atribuídas ao PRÓXIMO pregão
      (merge_asof forward). Assim não se perde informação e não se usa o futuro.
    - Várias notícias no mesmo pregão -> média do score + contagem.
    - Pregões sem notícia -> forward fill limitado a 3 dias; depois, neutro (0).

    Premissa: notícias datadas no dia t estão disponíveis até o fechamento de t
    (momento da decisão). Se a sua base tiver horário, desloque para t+1 as
    notícias publicadas após o fechamento do pregão para evitar vazamento.
    """
    s = sentimento.reset_index(drop=True).sort_values("Data")
    pregoes = pd.DataFrame({"Data_Pregao": precos.index}).sort_values("Data_Pregao")

    mapeado = pd.merge_asof(s, pregoes, left_on="Data", right_on="Data_Pregao",
                            direction="forward")
    diario = (mapeado.dropna(subset=["Data_Pregao"])
              .groupby("Data_Pregao")
              .agg(Sentimento_Score=("Sentimento_Score", "mean"),
                   N_Noticias=("Sentimento_Score", "size")))

    df = precos.join(diario, how="left")
    df["Sem_Noticia"] = df["Sentimento_Score"].isna().astype(int)
    df["Sentimento_Score"] = df["Sentimento_Score"].ffill(limit=3).fillna(0.0)
    df["N_Noticias"] = df["N_Noticias"].fillna(0.0)
    return df


# -----------------------------------------------------------------------------------
# 3. ENGENHARIA DE FEATURES + TARGET
# -----------------------------------------------------------------------------------
FEATURES_PRECO = [
    "Retorno_Log", "Amplitude", "Corpo",
    "MM5_Ratio", "MM10_Ratio", "MM20_Ratio", "MM50_Ratio",
    "Volatilidade_20", "RSI_14", "Volume_Var",
]
FEATURES_SENT = ["Sentimento_Score", "Sent_MM3", "Sent_Delta", "N_Noticias", "Sem_Noticia"]


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usamos features RELATIVAS (retornos, razão preço/média móvel, RSI...) em vez de
    preços brutos: um MinMaxScaler ajustado no treino gera valores fora de [0, 1]
    quando o preço do teste ultrapassa o máximo histórico do treino. Features
    relativas são mais estacionárias e generalizam melhor.
    Todas usam apenas informação até o dia t (rolling/diff olham para trás).
    """
    d = df.copy()
    c = d["Adj Close"]

    d["Retorno_Log"] = np.log(c).diff()
    d["Amplitude"] = (d["High"] - d["Low"]) / d["Close"]
    d["Corpo"] = (d["Close"] - d["Open"]) / d["Open"]

    # Médias móveis expressas como distância percentual do preço até a média
    for n in (5, 10, 20, 50):
        d[f"MM{n}_Ratio"] = c / c.rolling(n).mean() - 1

    d["Volatilidade_20"] = d["Retorno_Log"].rolling(20).std()

    # RSI de 14 períodos (média simples)
    delta = c.diff()
    ganho = delta.clip(lower=0).rolling(14).mean()
    perda = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - 100 / (1 + ganho / perda.replace(0, np.nan))

    d["Volume_Var"] = np.log(d["Volume"].replace(0, np.nan)).diff()

    # Dinâmica do sentimento
    d["Sent_MM3"] = d["Sentimento_Score"].rolling(3).mean()
    d["Sent_Delta"] = d["Sentimento_Score"].diff()

    # TARGET: 1 se fechamento de t+1 > fechamento de t (estritamente), senão 0
    prox = c.shift(-1)
    d["Target"] = np.where(prox > c, 1.0, 0.0)
    d.loc[prox.isna(), "Target"] = np.nan          # último dia: futuro desconhecido

    # Retorno do dia seguinte - usado SOMENTE no backtest, nunca como feature
    d["Retorno_Futuro"] = np.log(prox / c)

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES_PRECO + FEATURES_SENT)
    return d


# -----------------------------------------------------------------------------------
# 4. JANELAS TEMPORAIS
# -----------------------------------------------------------------------------------
def criar_janelas(X: np.ndarray, y: np.ndarray, janela: int, indices_fim: np.ndarray):
    """
    Converte o dataset multivariado em sequências 3D (amostras, janela, features).
    A janela que TERMINA no dia t (inclusive) prevê Target[t] = direção de t -> t+1.
    'indices_fim' define quais dias t viram amostra (permite separar treino/teste
    sem perder o histórico necessário para montar as primeiras janelas do teste).
    """
    Xs = np.stack([X[t - janela + 1: t + 1] for t in indices_fim])
    ys = y[indices_fim]
    return Xs.astype("float32"), ys.astype("float32")


# -----------------------------------------------------------------------------------
# 5. MODELO HÍBRIDO CNN-LSTM (dois ramos)
# -----------------------------------------------------------------------------------
def construir_modelo(janela: int, n_preco: int, n_sent: int) -> tf.keras.Model:
    """
    Ramo 1 (preço): Conv1D extrai padrões locais de curto prazo -> LSTM captura
                    dependências de longo prazo.
    Ramo 2 (sentimento): LSTM leve sobre o fluxo de sentimento.
    Os ramos são concatenados e passam por camadas densas até a saída sigmoide.
    """
    in_preco = layers.Input(shape=(janela, n_preco), name="preco")
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(in_preco)
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)

    in_sent = layers.Input(shape=(janela, n_sent), name="sentimento")
    s = layers.LSTM(16)(in_sent)
    s = layers.Dropout(0.3)(s)

    z = layers.Concatenate()([x, s])
    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    saida = layers.Dense(1, activation="sigmoid", name="direcao")(z)

    modelo = models.Model([in_preco, in_sent], saida, name="CNN_LSTM_Sentimento")
    modelo.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                   loss="binary_crossentropy",
                   metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return modelo


# -----------------------------------------------------------------------------------
# 6. AVALIAÇÃO
# -----------------------------------------------------------------------------------
def avaliar(y_true, prob, ret_futuro, datas, limiar=0.5):
    y_pred = (prob >= limiar).astype(int)

    print("\n" + "=" * 70)
    print(" CLASSIFICATION REPORT (conjunto de teste)")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=["Baixa (0)", "Alta (1)"],
                                digits=4, zero_division=0))

    acc_dir = accuracy_score(y_true, y_pred)
    baseline = max(y_true.mean(), 1 - y_true.mean())   # sempre chutar a classe majoritária
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = float("nan")

    print(f" >>> ACURÁCIA DIRECIONAL: {acc_dir:.2%}")
    print(f"     Baseline ingênuo (classe majoritária): {baseline:.2%}")
    print(f"     Ganho sobre o baseline: {(acc_dir - baseline) * 100:+.2f} p.p.")
    print(f"     ROC-AUC: {auc:.4f}")

    # Mini-backtest: comprado quando prevê alta, fora do mercado quando prevê baixa
    # (sem custos de corretagem/slippage - serve só como referência)
    ret_estrategia = np.where(y_pred == 1, ret_futuro, 0.0)
    acum_estrat = np.exp(np.cumsum(ret_estrategia)) - 1
    acum_bh = np.exp(np.cumsum(ret_futuro)) - 1
    print(f"     Retorno acumulado estratégia: {acum_estrat[-1]:.2%} | "
          f"Buy & Hold: {acum_bh[-1]:.2%}")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Baixa", "Alta"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title(f"Matriz de Confusão - Acurácia Direcional {acc_dir:.2%}")

    axes[1].plot(datas, acum_bh * 100, label="Buy & Hold")
    axes[1].plot(datas, acum_estrat * 100, label="Estratégia do modelo")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("Retorno acumulado no teste (%)")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plotar_treino(hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist.history[m], label="treino")
        ax.plot(hist.history[f"val_{m}"], label="validação")
        ax.set_title(m)
        ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------
# 7. SALVAMENTO DE ARTEFATOS (dataset + modelo .pth)
# -----------------------------------------------------------------------------------
def salvar_dataset(dados: pd.DataFrame, sentimento: pd.DataFrame, cfg) -> None:
    """
    Salva duas versões do dataset utilizado:
      - dataset_<ticker>.csv: preços + sentimento alinhado + features + Target, com a
        coluna 'Conjunto' indicando o papel de cada dia (treino, validacao, teste...).
      - sentimento_usado.csv: o sentimento bruto no formato que o pipeline espera
        (Data, Sentimento_Score). Renomeie para 'sentimento.csv' para reproduzir
        exatamente a mesma execução, inclusive com os dados sintéticos.
    """
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    nome = cfg["ticker"].replace("^", "").replace(".", "_")
    caminho = os.path.join(cfg["pasta_saida"], f"dataset_{nome}.csv")
    dados.to_csv(caminho, index_label="Data", float_format="%.8f")

    caminho_sent = os.path.join(cfg["pasta_saida"], "sentimento_usado.csv")
    sentimento.reset_index(drop=True)[["Data", "Sentimento_Score"]].to_csv(
        caminho_sent, index=False, float_format="%.8f")

    print(f"[SALVO] Dataset: {caminho} ({len(dados)} linhas, {dados.shape[1]} colunas)")
    print(f"[SALVO] Sentimento: {caminho_sent}")


def salvar_modelo_pth(modelo: tf.keras.Model, scaler: MinMaxScaler, cfg) -> str:
    """
    Salva o modelo em .pth (formato do PyTorch, via torch.save).

    Observação: o modelo foi treinado em Keras, então o .pth guarda um checkpoint
    com os pesos convertidos em tensores PyTorch + tudo que é necessário para
    reconstruí-lo (janela, listas de features e parâmetros do scaler). Use
    carregar_modelo_pth() para restaurar. O arquivo pode ser aberto com
    torch.load(..., weights_only=True), que é o modo seguro do PyTorch.

    Também salva uma cópia no formato nativo do Keras (.keras), que é o mais
    simples de recarregar com tf.keras.models.load_model().
    """
    import torch  # pip install torch (gratuito); import tardio para não pesar no treino

    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    pesos = modelo.get_weights()
    checkpoint = {
        "formato": "keras_pesos_em_tensores_torch",
        "nome_modelo": modelo.name,
        "pesos": {f"w_{i:03d}": torch.from_numpy(np.ascontiguousarray(w))
                  for i, w in enumerate(pesos)},
        "nomes_pesos": [v.path if hasattr(v, "path") else v.name for v in modelo.weights],
        "janela": int(cfg["janela"]),
        "features_preco": list(FEATURES_PRECO),
        "features_sent": list(FEATURES_SENT),
        "ticker": cfg["ticker"],
        "scaler": {
            "min_": torch.from_numpy(scaler.min_.copy()),
            "scale_": torch.from_numpy(scaler.scale_.copy()),
            "data_min_": torch.from_numpy(scaler.data_min_.copy()),
            "data_max_": torch.from_numpy(scaler.data_max_.copy()),
            "data_range_": torch.from_numpy(scaler.data_range_.copy()),
            "n_samples_seen_": int(scaler.n_samples_seen_),
        },
    }
    caminho_pth = os.path.join(cfg["pasta_saida"], "modelo_cnn_lstm.pth")
    torch.save(checkpoint, caminho_pth)

    caminho_keras = os.path.join(cfg["pasta_saida"], "modelo_cnn_lstm.keras")
    modelo.save(caminho_keras)

    print(f"[SALVO] Modelo (.pth): {caminho_pth}")
    print(f"[SALVO] Modelo (.keras): {caminho_keras}")
    return caminho_pth


def carregar_modelo_pth(caminho: str):
    """Reconstrói o modelo Keras e o scaler a partir do checkpoint .pth."""
    import torch

    ck = torch.load(caminho, weights_only=True)
    modelo = construir_modelo(ck["janela"], len(ck["features_preco"]), len(ck["features_sent"]))
    modelo.set_weights([ck["pesos"][f"w_{i:03d}"].numpy() for i in range(len(ck["pesos"]))])

    s = ck["scaler"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    for atributo in ("min_", "scale_", "data_min_", "data_max_", "data_range_"):
        setattr(scaler, atributo, s[atributo].numpy())
    scaler.n_samples_seen_ = s["n_samples_seen_"]
    scaler.n_features_in_ = len(s["min_"])
    scaler.feature_names_in_ = np.array(ck["features_preco"] + ck["features_sent"], dtype=object)

    print(f"[INFO] Modelo carregado de {caminho} (janela={ck['janela']}, ticker={ck['ticker']})")
    return modelo, scaler, ck


# -----------------------------------------------------------------------------------
# 8. PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------------
def main(cfg=CONFIG):
    fixar_sementes(cfg["seed"])
    J = cfg["janela"]

    # 1) Aquisição + merge
    precos = baixar_precos(cfg["ticker"], cfg["inicio"], cfg["fim"])
    sentimento = carregar_sentimento(cfg["csv_sentimento"], precos, cfg["seed"])
    dados = criar_features(alinhar_dados(precos, sentimento))
    print(f"[INFO] {cfg['ticker']}: {len(dados)} pregões de "
          f"{dados.index[0].date()} a {dados.index[-1].date()}")

    # Separa o último dia (Target desconhecido) para a previsão de "amanhã"
    rotulados = dados.dropna(subset=["Target"])
    colunas = FEATURES_PRECO + FEATURES_SENT
    n = len(rotulados)

    # 2) Divisão cronológica (sem shuffle) + purga na fronteira
    #    O Target do último dia de treino usa o fechamento do 1º dia de teste;
    #    por isso esse dia é descartado do treino (purga de 1 pregão).
    corte = int(n * cfg["prop_treino"])
    fim_treino = corte - 2                 # último índice de treino (após purga)

    # 3) Scaler ajustado SOMENTE no treino
    scaler = MinMaxScaler()
    scaler.fit(rotulados[colunas].iloc[: fim_treino + 1])
    X_all = scaler.transform(rotulados[colunas])
    y_all = rotulados["Target"].values

    # 4) Janelas. As primeiras janelas do teste usam features (não targets) dos
    #    últimos 59 dias do treino, o que é legítimo: esse histórico já é conhecido.
    idx_treino_total = np.arange(J - 1, fim_treino + 1)
    idx_teste = np.arange(corte, n)

    # Validação = fatia final do treino (cronológica), com purga de 1 amostra
    n_val = int(len(idx_treino_total) * cfg["prop_validacao"])
    idx_treino = idx_treino_total[: -n_val - 1]
    idx_val = idx_treino_total[-n_val:]

    # Coluna 'Conjunto' para o dataset salvo: mostra o papel de cada pregão
    papel = np.full(n, "purga", dtype=object)
    papel[: J - 1] = "historico_janela"
    papel[idx_treino] = "treino"
    papel[idx_val] = "validacao"
    papel[idx_teste] = "teste"
    conjunto = pd.Series("previsao_amanha", index=dados.index, dtype=object)
    conjunto.loc[rotulados.index] = papel
    dados = dados.assign(Conjunto=conjunto)

    Xtr, ytr = criar_janelas(X_all, y_all, J, idx_treino)
    Xva, yva = criar_janelas(X_all, y_all, J, idx_val)
    Xte, yte = criar_janelas(X_all, y_all, J, idx_teste)

    n_p = len(FEATURES_PRECO)
    dividir = lambda X: [X[:, :, :n_p], X[:, :, n_p:]]   # ramo preço | ramo sentimento

    print(f"[INFO] Janelas -> treino {Xtr.shape} | validação {Xva.shape} | teste {Xte.shape}")
    print(f"[INFO] Proporção de altas no treino: {ytr.mean():.2%}")

    # 5) Modelo
    modelo = construir_modelo(J, n_p, len(FEATURES_SENT))
    modelo.summary()

    # Pesos de classe para compensar desbalanceamento entre altas e baixas
    p1 = ytr.mean()
    pesos = {0: 0.5 / (1 - p1), 1: 0.5 / p1}

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    hist = modelo.fit(dividir(Xtr), ytr,
                      validation_data=(dividir(Xva), yva),
                      epochs=cfg["epocas"], batch_size=cfg["batch"],
                      class_weight=pesos, callbacks=cbs, verbose=1)
    plotar_treino(hist)

    # 6) Avaliação no teste (dados nunca vistos)
    prob = modelo.predict(dividir(Xte), verbose=0).ravel()
    ret_fut = rotulados["Retorno_Futuro"].values[idx_teste]
    datas = rotulados.index[idx_teste]
    avaliar(yte.astype(int), prob, ret_fut, datas)

    # 7) Previsão para o próximo pregão (usa os últimos 60 dias disponíveis)
    X_ult = scaler.transform(dados[colunas].iloc[-J:]).astype("float32")[None, ...]
    p_amanha = float(modelo.predict(dividir(X_ult), verbose=0)[0, 0])
    print(f"\n[PREVISÃO] Após {dados.index[-1].date()}: P(alta) = {p_amanha:.2%} -> "
          f"{'ALTA' if p_amanha >= 0.5 else 'BAIXA'}")

    # 8) Salvamento: dataset utilizado + modelo em .pth (e .keras)
    salvar_dataset(dados, sentimento, cfg)
    try:
        salvar_modelo_pth(modelo, scaler, cfg)
    except ImportError:
        print("[AVISO] PyTorch não instalado: rode 'pip install torch' para gerar o .pth. "
              "Salvando apenas no formato .keras.")
        os.makedirs(cfg["pasta_saida"], exist_ok=True)
        modelo.save(os.path.join(cfg["pasta_saida"], "modelo_cnn_lstm.keras"))

    return modelo, scaler, hist


if __name__ == "__main__":
    main()