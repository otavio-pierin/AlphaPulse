"""
Prepara uma ou mais ações para a aplicação web: coleta notícias, calcula o
sentimento com FinBERT, treina a CNN-LSTM e salva o modelo (.pth).

Uso:
    python preparar_ativos.py --listar
    python preparar_ativos.py PETR4.SA VALE3.SA
    python preparar_ativos.py --todos

Cada ação leva de 20 a 30 minutos na primeira vez (a maior parte é a coleta de
notícias). Se for interrompido, rode de novo: a coleta retoma de onde parou.
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")          # sem janelas de gráfico durante o treino em lote

import main as M
from ativos import ATIVOS, caminho_modelo, config_ativo


def preparar(ticker: str) -> None:
    cfg = config_ativo(ticker)
    print("\n" + "=" * 70)
    print(f" Preparando {ATIVOS[ticker]['nome']} ({ticker})")
    print("=" * 70)
    M.main(cfg)
    print(f"[OK] Modelo salvo em {caminho_modelo(cfg)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treina os modelos usados pela aplicação web.")
    ap.add_argument("tickers", nargs="*", help="ex.: PETR4.SA VALE3.SA")
    ap.add_argument("--todos", action="store_true", help="prepara todos os ativos do catálogo")
    ap.add_argument("--listar", action="store_true", help="mostra os ativos e se já têm modelo")
    a = ap.parse_args()

    if a.listar or (not a.tickers and not a.todos):
        for t, info in ATIVOS.items():
            pronto = os.path.exists(caminho_modelo(config_ativo(t)))
            print(f"  {t:10s} {info['nome']:22s} {'modelo pronto' if pronto else 'sem modelo'}")
        raise SystemExit

    alvos = list(ATIVOS) if a.todos else a.tickers
    invalidos = [t for t in alvos if t not in ATIVOS]
    if invalidos:
        raise SystemExit(f"Ativos fora do catálogo: {invalidos}. Adicione-os em ativos.py.")
    for t in alvos:
        preparar(t)
