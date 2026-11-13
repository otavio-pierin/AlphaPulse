"""
Gera a base de dados do AlphaPulse:
  1. pregões do ativo (yfinance)
  2. notícias reais (Google News + Suno Notícias)
  3. sentimento de cada manchete (FinBERT-PT-BR)
  4. sentimento agregado por pregão + junção com os preços

Uso:
    python gerar_base.py
    python gerar_base.py --ticker PETR4.SA --consulta "Petrobras OR PETR4" --termos Petrobras PETR4
    python gerar_base.py --so-pregoes
    python gerar_base.py --sem-google        (usa só a API do Suno)

Saídas em dados/:
    pregoes_<ativo>.csv                 OHLCV diário
    noticias_<ativo>.csv                manchetes com probabilidades e score
    base_diaria_<ativo>.csv             pregões + sentimento agregado (entrada do modelo)
"""

import argparse
import copy
import os

import pandas as pd

from banco import Banco
from config import CONFIG
from noticias import coletar_noticias
from pregoes import baixar_pregoes, resumo_pregoes
from sentimento import agregar_por_pregao, pontuar_pendentes


def nome_ativo(ticker):
    return ticker.replace("^", "").replace(".SA", "").replace(".", "_")


def salvar_csv(df, caminho, **kw):
    tmp = caminho + ".tmp"
    df.to_csv(tmp, **kw)
    os.replace(tmp, caminho)             # troca atômica: nunca fica um CSV pela metade


def main(cfg):
    os.makedirs(cfg["pasta_dados"], exist_ok=True)
    ativo = nome_ativo(cfg["ticker"])

    # 1) Pregões
    pregoes = baixar_pregoes(cfg["ticker"], cfg["inicio"], cfg["fim"], cfg["fuso"])
    resumo_pregoes(pregoes, cfg["ticker"])
    caminho = os.path.join(cfg["pasta_dados"], f"pregoes_{ativo}.csv")
    salvar_csv(pregoes, caminho, float_format="%.6f")
    print(f"[SALVO] {caminho}")
    if cfg.get("so_pregoes"):
        return

    # Cada ativo tem seu banco: as consultas de notícias são diferentes
    banco = Banco(cfg["banco"].replace(".db", f"_{ativo}.db"))
    try:
        # 2) Notícias (começa 7 dias antes para cobrir o fim de semana anterior ao 1º pregão)
        inicio = pregoes.index.min() - pd.Timedelta(days=7)
        fim = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        coletar_noticias(banco, cfg, inicio, fim)

        # 3) Sentimento
        pontuar_pendentes(banco, cfg)
        noticias = banco.ler_noticias()
    finally:
        banco.fechar()

    caminho = os.path.join(cfg["pasta_dados"], f"noticias_{ativo}.csv")
    salvar_csv(noticias, caminho, index=False, float_format="%.6f")
    print(f"[SALVO] {caminho} ({len(noticias)} manchetes)")

    # 4) Agregação por pregão + junção
    diario = agregar_por_pregao(noticias, pregoes.index)
    base = pregoes.join(diario)
    caminho = os.path.join(cfg["pasta_dados"], f"base_diaria_{ativo}.csv")
    salvar_csv(base, caminho, float_format="%.6f")
    print(f"[SALVO] {caminho} ({len(base)} pregões, {base.shape[1]} colunas)")

    # Resumo da cobertura
    print("\n[cobertura] Notícias por fonte:")
    print(noticias.groupby(["fonte", "veiculo"]).size().sort_values(ascending=False)
          .head(10).to_string())
    cob = base.groupby(base.index.year)["Sem_Noticia"].agg(lambda s: 1 - s.mean())
    print("\n[cobertura] Pregões com pelo menos uma notícia, por ano:")
    print((cob * 100).round(1).astype(str).add("%").to_string())
    print(f"\n[sentimento] Score médio {noticias['score'].mean():+.3f} | "
          f"positivas {(noticias['score'] > 0.15).mean():.0%} | "
          f"negativas {(noticias['score'] < -0.15).mean():.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gera a base de pregões e sentimento do AlphaPulse.")
    ap.add_argument("--ticker")
    ap.add_argument("--inicio")
    ap.add_argument("--consulta", help="consulta do Google News")
    ap.add_argument("--termos", nargs="+", help="termos de busca nos portais WordPress")
    ap.add_argument("--so-pregoes", action="store_true")
    ap.add_argument("--sem-google", action="store_true")
    ap.add_argument("--sem-wordpress", action="store_true")
    a = ap.parse_args()

    cfg = copy.deepcopy(CONFIG)
    if a.ticker:
        cfg["ticker"] = a.ticker
    if a.inicio:
        cfg["inicio"] = a.inicio
    if a.consulta:
        cfg["google_news"]["consulta"] = a.consulta
    if a.termos:
        for site in cfg["wordpress"]["sites"]:
            site["termos"] = a.termos
    cfg["google_news"]["ativo"] &= not a.sem_google
    cfg["wordpress"]["ativo"] &= not a.sem_wordpress
    cfg["so_pregoes"] = a.so_pregoes
    main(cfg)