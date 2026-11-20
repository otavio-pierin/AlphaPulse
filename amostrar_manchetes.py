"""
Passo 1 da validação em manchetes REAIS do produto: sorteia a amostra para rotulagem manual.

Por que: os ~88% do FinBERT ajustado foram medidos em notícias da Unicamp (2017-2019). O app,
porém, pontua manchetes de 2018 até hoje, vindas do Google News e do Suno. Esta validação
mede se o modelo continua bom nas manchetes que o produto realmente exibe.

Como funciona:
  1. python amostrar_manchetes.py
     -> dados/validacao_manchetes_A.csv e dados/validacao_manchetes_B.csv
        (as MESMAS manchetes, embaralhadas, SEM o score do modelo, para não enviesar quem rotula)
  2. Duas pessoas do grupo preenchem a coluna `rotulo`, cada uma no seu arquivo e sem conversar:
        pos = a notícia é boa para o mercado/Ibovespa
        neg = a notícia é ruim para o mercado/Ibovespa
        neu = sem efeito claro, ou o efeito depende de outros fatores
     Dica: julguem o efeito sobre a BOLSA, não o tom da frase ("dólar cai" costuma ser pos).
  3. python validar_manchetes_recentes.py dados/validacao_manchetes_A.csv dados/validacao_manchetes_B.csv

A amostra é estratificada por ano (mesmo número de manchetes por ano), o que permite ver se o
desempenho cai nos anos mais distantes do treino ou nos mais recentes.

Uso:
    python amostrar_manchetes.py                  (150 manchetes)
    python amostrar_manchetes.py --n 200 --seed 7
"""

import argparse
import math
import os

import pandas as pd

from config import CONFIG
from servico_app import caminhos


def main():
    ap = argparse.ArgumentParser(description="Sorteia manchetes para rotulagem manual.")
    ap.add_argument("--n", type=int, default=150, help="total de manchetes (padrão: 150)")
    ap.add_argument("--seed", type=int, default=42, help="semente do sorteio (reprodutível)")
    ap.add_argument("--saida", default=CONFIG["pasta_dados"], help="pasta de saída")
    a = ap.parse_args()

    caminho = caminhos()["noticias"]
    if not os.path.exists(caminho):
        raise SystemExit(f"Não achei {caminho}. Rode antes: python gerar_base.py")

    n = pd.read_csv(caminho)
    n["data_efetiva"] = pd.to_datetime(n["data_efetiva"])
    n = n.dropna(subset=["titulo", "score"]).drop_duplicates("id")
    n["ano"] = n["data_efetiva"].dt.year

    anos = sorted(n["ano"].unique())
    por_ano = math.ceil(a.n / len(anos))
    amostra = pd.concat([g.sample(min(len(g), por_ano), random_state=a.seed)
                         for _, g in n.groupby("ano")])
    if len(amostra) > a.n:                                   # corta o excesso ao acaso
        amostra = amostra.sample(a.n, random_state=a.seed)
    amostra = amostra.sample(frac=1, random_state=a.seed)    # embaralha: a ordem não revela o ano

    saida = amostra[["id", "data_efetiva", "veiculo", "titulo"]].copy()
    saida["data_efetiva"] = saida["data_efetiva"].dt.strftime("%Y-%m-%d")
    saida = saida.rename(columns={"data_efetiva": "data"})
    saida["rotulo"] = ""

    os.makedirs(a.saida, exist_ok=True)
    for pessoa in ("A", "B"):
        destino = os.path.join(a.saida, f"validacao_manchetes_{pessoa}.csv")
        # ';' e BOM utf-8: abre direto no Excel em português
        saida.to_csv(destino, index=False, sep=";", encoding="utf-8-sig")
        print(f"[SALVO] {destino} ({len(saida)} manchetes)")

    print("\nManchetes por ano:", amostra["ano"].value_counts().sort_index().to_dict())
    print("\nPreencham a coluna 'rotulo' com pos, neg ou neu (cada pessoa no seu arquivo).")
    print("Depois: python validar_manchetes_recentes.py "
          f"{os.path.join(a.saida, 'validacao_manchetes_A.csv')} "
          f"{os.path.join(a.saida, 'validacao_manchetes_B.csv')}")


if __name__ == "__main__":
    main()
