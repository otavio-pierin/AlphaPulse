"""
Catálogo de ativos disponíveis na aplicação e configuração do pipeline para cada um.

Cada ativo tem seu próprio modelo, base de notícias e artefatos, porque as
notícias relevantes (e o comportamento do preço) mudam de uma ação para outra.
Para incluir uma ação, basta adicionar uma entrada em ATIVOS e rodar:
    python preparar_ativos.py TICKER
"""

import copy
import os

from main import CONFIG

# consulta = busca no Google News (aceita OR e aspas para termos exatos)
ATIVOS = {
    "^BVSP":    {"nome": "Ibovespa",            "consulta": 'Ibovespa OR "Bolsa brasileira"', "legado": True},
    "PETR4.SA": {"nome": "Petrobras PN",        "consulta": 'Petrobras OR PETR4'},
    "VALE3.SA": {"nome": "Vale ON",             "consulta": '"mineradora Vale" OR VALE3'},
    "ITUB4.SA": {"nome": "Itaú Unibanco PN",    "consulta": '"Itaú Unibanco" OR ITUB4'},
    "BBAS3.SA": {"nome": "Banco do Brasil ON",  "consulta": '"Banco do Brasil" OR BBAS3'},
    "BBDC4.SA": {"nome": "Bradesco PN",         "consulta": 'Bradesco OR BBDC4'},
    "ABEV3.SA": {"nome": "Ambev ON",            "consulta": 'Ambev OR ABEV3'},
    "WEGE3.SA": {"nome": "WEG ON",              "consulta": 'WEGE3 OR "WEG S.A."'},
}


def slug(ticker: str) -> str:
    return ticker.replace("^", "").replace(".SA", "").replace(".", "_")


def config_ativo(ticker: str) -> dict:
    """Copia o CONFIG do main.py e ajusta ticker, consulta de notícias e pastas."""
    info = ATIVOS[ticker]
    cfg = copy.deepcopy(CONFIG)
    cfg["ticker"] = ticker
    cfg["consulta_google_news"] = info["consulta"]
    cfg["fontes_noticias"] = ("google_news",)      # GDELT bloqueia com frequência

    if info.get("legado"):
        # O Ibovespa reaproveita o que já foi gerado pelo main.py (sem retreinar)
        cfg.update(csv_sentimento="sentimento.csv", pasta_noticias="dados_noticias",
                   pasta_saida="artefatos")
    else:
        base = os.path.join("dados_ativos", slug(ticker))
        cfg.update(csv_sentimento=os.path.join(base, "sentimento.csv"),
                   pasta_noticias=os.path.join(base, "noticias"),
                   pasta_saida=os.path.join(base, "artefatos"))
    return cfg


def caminho_modelo(cfg: dict) -> str:
    return os.path.join(cfg["pasta_saida"], "modelo_cnn_lstm.pth")


def caminho_noticias(cfg: dict) -> str:
    return os.path.join(cfg["pasta_noticias"], "noticias_com_sentimento.csv")
