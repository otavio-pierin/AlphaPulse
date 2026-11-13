"""
Extração dos dados diários dos pregões com yfinance.
"""

import time

import numpy as np
import pandas as pd
import yfinance as yf

COLUNAS = {"Open": "Abertura", "High": "Maxima", "Low": "Minima", "Close": "Fechamento",
           "Adj Close": "Fechamento_Ajustado", "Volume": "Volume"}


def baixar_pregoes(ticker, inicio, fim=None, fuso="America/Sao_Paulo", tentativas=3):
    """
    Baixa OHLCV diário e devolve um DataFrame indexado pela data do pregão.

    - Durante o horário de negociação o yfinance inclui o candle do dia ainda
      incompleto; ele é descartado até as 18h (Brasília).
    - As datas ficam sem fuso e em nanossegundos, para não haver conflito de
      tipos ao juntar com as notícias.
    """
    df = pd.DataFrame()
    for i in range(tentativas):
        df = yf.download(ticker, start=inicio, end=fim, auto_adjust=False,
                         progress=False, threads=False)
        if not df.empty:
            break
        time.sleep(3 * (i + 1))
    if df.empty:
        raise RuntimeError(f"O yfinance não retornou dados para {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):           # versões recentes: (campo, ticker)
        df.columns = df.columns.get_level_values(0)
    df = df[list(COLUNAS)].rename(columns=COLUNAS)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index).tz_localize(None).normalize()).as_unit("ns")
    df.index.name = "Data"
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Fechamento"]).sort_index()

    agora = pd.Timestamp.now(tz=fuso)
    if df.index[-1].date() == agora.date() and agora.hour < 18:
        df = df.iloc[:-1]

    df["Retorno_Log"] = np.log(df["Fechamento_Ajustado"]).diff()
    df["Variacao_Pct"] = df["Fechamento"].pct_change() * 100
    return df


def resumo_pregoes(df, ticker):
    u = df.iloc[-1]
    print(f"[pregões] {ticker}: {len(df)} pregões de {df.index[0]:%d/%m/%Y} a {df.index[-1]:%d/%m/%Y}")
    print(f"[pregões] Último fechamento: {u['Fechamento']:,.2f} ({u['Variacao_Pct']:+.2f}%)")