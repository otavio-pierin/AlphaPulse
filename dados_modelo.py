"""
Prepara os dados dos modelos a partir de dados/base_diaria_<ativo>.csv (gerar_base.py).

Ramos de entrada:
  preço  -> indicadores técnicos (Modelo 1 da late fusion)
  texto  -> sentimento do FinBERT agregado por pregão (Modelo 2 da late fusion)
  early  -> preço + texto concatenados dia a dia, numa única sequência (early fusion)

Avaliação walk-forward: em vez de um único corte, o período final é dividido em
vários blocos de teste consecutivos (folds). Em cada fold, os modelos são treinados
só com o passado daquele bloco (janela expansível):

  fold 1: [ treino ............ ][val][meta][teste 1]
  fold 2: [ treino ..................... ][val][meta][teste 2]
  fold 3: [ treino .............................. ][val][meta][teste 3]

Entre blocos são descartados 'horizonte' dias (purga), porque o alvo de um dia usa o
fechamento de 'horizonte' pregões à frente.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES_PRECO = ["Retorno_Log", "Amplitude", "Corpo", "MM5_Ratio", "MM10_Ratio", "MM20_Ratio",
                  "MM50_Ratio", "Volatilidade_20", "RSI_14", "Volume_Var"]
LAGS_RETORNO = [f"Retorno_Lag{k}" for k in range(1, 6)]
# Ramo de texto simplificado: antes eram 10 features, várias quase idênticas entre si,
# o que levava a coeficientes contraditórios. Ficam o sentimento do dia e sua média de 5 dias.
FEATURES_TEXTO = ["Sent_Ponderado", "Sent_MM5"]

# Alvo de risco: volatilidade de curto prazo e volume de notícias ajudam a antecipar
# dias agitados (a volatilidade se agrupa no tempo)
FEATURES_PRECO_RISCO = FEATURES_PRECO + ["Abs_Retorno", "Volatilidade_5", "Razao_Vol"]
FEATURES_TEXTO_RISCO = FEATURES_TEXTO + ["Log_N_Noticias", "Prop_Negativas"]
JANELA_REF_RISCO = 60          # pregões usados na mediana de referência do risco


def criar_features(base: pd.DataFrame, horizonte: int = 1, alvo: str = "direcao") -> pd.DataFrame:
    """
    Features só com informação até o fechamento de t.
    Alvos:
      direcao -> 1 se Fechamento(t+h) > Fechamento(t)
      risco   -> 1 se a oscilação média dos próximos h pregões (|retorno|) for maior que a
                 mediana dessa mesma medida nos últimos 60 pregões (dia de risco alto)
    """
    d = base.copy()
    c = d["Fechamento_Ajustado"]
    d["Retorno_Log"] = np.log(c).diff()
    d["Amplitude"] = (d["Maxima"] - d["Minima"]) / d["Fechamento"]
    d["Corpo"] = (d["Fechamento"] - d["Abertura"]) / d["Abertura"]
    for n in (5, 10, 20, 50):
        d[f"MM{n}_Ratio"] = c / c.rolling(n).mean() - 1
    d["Volatilidade_20"] = d["Retorno_Log"].rolling(20).std()
    delta = c.diff()
    ganho, perda = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - 100 / (1 + ganho / perda.replace(0, np.nan))
    d["Volume_Var"] = np.log(d["Volume"].replace(0, np.nan)).diff()
    for k in range(1, 6):
        d[f"Retorno_Lag{k}"] = d["Retorno_Log"].shift(k)
    d["Sent_MM5"] = d["Sent_Ponderado"].rolling(5).mean()
    d["Abs_Retorno"] = d["Retorno_Log"].abs()
    d["Volatilidade_5"] = d["Retorno_Log"].rolling(5).std()
    d["Razao_Vol"] = d["Volatilidade_5"] / d["Volatilidade_20"]
    d["Log_N_Noticias"] = np.log1p(d["N_Noticias"])

    futuro = c.shift(-horizonte)
    d["Retorno_Futuro"] = np.log(futuro / c)             # só para o backtest
    if alvo == "direcao":
        d["Target"] = np.where(futuro > c, 1.0, 0.0)
        d.loc[futuro.isna(), "Target"] = np.nan
    elif alvo == "risco":
        osc = d["Abs_Retorno"]
        # oscilação média dos próximos h pregões (t+1 ... t+h)
        osc_futura = osc[::-1].rolling(horizonte).mean()[::-1].shift(-1)
        # referência: mediana da mesma medida no passado (até t), sem olhar o futuro
        osc_passada = osc.rolling(horizonte).mean()
        d["Ref_Risco"] = osc_passada.rolling(JANELA_REF_RISCO).median()
        d["Osc_Futura"] = osc_futura
        d["Target"] = np.where(osc_futura > d["Ref_Risco"], 1.0, 0.0)
        d.loc[osc_futura.isna(), "Target"] = np.nan
    else:
        raise ValueError("alvo deve ser 'direcao' ou 'risco'")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES_PRECO_RISCO + LAGS_RETORNO + FEATURES_TEXTO_RISCO)
    if alvo == "risco":
        d = d[d["Ref_Risco"].notna()]                     # começo da série sem referência
    return d


@dataclass
class Base:
    df: pd.DataFrame           # todos os pregões; os 'horizonte' últimos não têm alvo
    n: int                     # quantidade de pregões com alvo
    datas: pd.DatetimeIndex
    y: np.ndarray
    ret_futuro: np.ndarray
    janela: int
    horizonte: int
    alvo: str
    feat_preco: list
    feat_texto: list


def carregar(caminho_base, janela=30, horizonte=1, alvo="direcao"):
    d = criar_features(pd.read_csv(caminho_base, index_col="Data", parse_dates=True), horizonte, alvo)
    # Os 'horizonte' últimos pregões não têm alvo. Eles continuam na tabela (as janelas
    # precisam de dias consecutivos) e o último deles é usado na previsão final.
    n = int(d["Target"].notna().sum())
    assert d["Target"].iloc[:n].notna().all(), "alvos ausentes fora do fim da série"
    return Base(df=d, n=n, datas=d.index[:n], y=d["Target"].values[:n].astype("float32"),
                ret_futuro=d["Retorno_Futuro"].values[:n], janela=janela, horizonte=horizonte,
                alvo=alvo,
                feat_preco=FEATURES_PRECO_RISCO if alvo == "risco" else FEATURES_PRECO,
                feat_texto=FEATURES_TEXTO_RISCO if alvo == "risco" else FEATURES_TEXTO)


def gerar_folds(b: Base, n_folds=5, tam_teste=126, tam_val=126, tam_meta=126):
    """Folds walk-forward com janela de treino expansível. 126 pregões ~ 6 meses."""
    h, inicio = b.horizonte, b.janela - 1
    folds = []
    for k in range(n_folds):
        t1 = b.n - (n_folds - 1 - k) * tam_teste
        t0 = t1 - tam_teste
        m0 = t0 - tam_meta
        v0 = m0 - tam_val
        if v0 - h <= inicio + 250:
            raise ValueError("Folds demais para o tamanho da base: reduza --folds ou --tam-teste.")
        folds.append({"treino": np.arange(inicio, v0 - h), "val": np.arange(v0, m0 - h),
                      "meta": np.arange(m0, t0 - h), "teste": np.arange(t0, t1)})
    return folds


def _janelas(matriz, janela):
    """Janela que termina em t (inclusive) para todo t; posições iniciais ficam com zeros."""
    n, f = matriz.shape
    saida = np.zeros((n, janela, f), dtype="float32")
    for t in range(janela - 1, n):
        saida[t] = matriz[t - janela + 1: t + 1]
    return saida


@dataclass
class DadosFold:
    idx: dict
    X_tab: np.ndarray          # (pregões, F)   preço do dia + lags  -> MLP, SVM, regressão
    X_seq: np.ndarray          # (pregões, J, F) janelas de preço    -> LSTM, Bi-LSTM, CNN, Transformer
    X_texto: np.ndarray        # (pregões, Ft)  sentimento            -> modelo de texto
    X_early: np.ndarray        # (pregões, J, F+Ft) preço + texto     -> early fusion
    # a última linha de cada matriz é o pregão mais recente (usada na previsão final)


def montar_fold(b: Base, idx: dict) -> DadosFold:
    """Normalização ajustada SOMENTE nos pregões de treino deste fold."""
    tr = idx["treino"]
    fim_treino = tr[-1] + 1
    tab_cols = b.feat_preco + LAGS_RETORNO
    tab = StandardScaler().fit(b.df[tab_cols].iloc[tr]).transform(b.df[tab_cols])
    preco = StandardScaler().fit(b.df[b.feat_preco].iloc[:fim_treino]).transform(b.df[b.feat_preco])
    texto = StandardScaler().fit(b.df[b.feat_texto].iloc[:fim_treino]).transform(b.df[b.feat_texto])
    return DadosFold(
        idx=idx,
        X_tab=tab.astype("float32"),
        X_seq=_janelas(preco.astype("float32"), b.janela),
        X_texto=texto.astype("float32"),
        X_early=_janelas(np.hstack([preco, texto]).astype("float32"), b.janela),
    )


def resumo(b: Base, folds):
    classe = "dias de risco alto" if b.alvo == "risco" else "altas"
    print(f"[dados] alvo {b.alvo} | {b.n} pregões | janela {b.janela} | horizonte {b.horizonte} dia(s) | "
          f"{classe} {b.y.mean():.1%}")
    for k, f in enumerate(folds, 1):
        t = f["teste"]
        print(f"   fold {k}: treino {len(f['treino']):4d} | val {len(f['val'])} | meta {len(f['meta'])} | "
              f"teste {b.datas[t[0]]:%d/%m/%Y} a {b.datas[t[-1]]:%d/%m/%Y} (classe 1: {b.y[t].mean():.1%})")