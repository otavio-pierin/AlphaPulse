"""
Configuração central do AlphaPulse. Todos os módulos leem daqui.
"""

CONFIG = {
    # --- Ativo e período ---
    "ticker": "^BVSP",                 # Ibovespa. Ações: "PETR4.SA", "VALE3.SA"...
    "inicio": "2018-01-01",
    "fim": None,                       # None = até hoje

    # --- Armazenamento ---
    "pasta_dados": "dados",
    "banco": "dados/alphapulse.db",    # SQLite: notícias, sentimento e checkpoints

    # --- Regras de tempo (prevenção de vazamento) ---
    "fuso": "America/Sao_Paulo",
    "hora_corte": 17,                  # notícia publicada a partir das 17h conta para o dia seguinte

    # --- Fonte 1: Google News RSS (histórico por janelas de data) ---
    "google_news": {
        "ativo": True,
        "consulta": 'Ibovespa OR "Bolsa brasileira"',
        "passo_dias": 7,               # janela inicial; é dividida se atingir o limite
        "limite": 100,                 # máximo de itens por consulta do Google News
        "pausa": 1.5,                  # segundos entre requisições
    },

    # --- Fonte 2: portais com API pública do WordPress ---
    # Para adicionar outro portal, teste no navegador: <site>/wp-json/wp/v2/posts
    "wordpress": {
        "ativo": True,
        "passo_dias": 31,              # uma janela por mês, paginada de 100 em 100
        "pausa": 0.5,
        "sites": [
            {
                "nome": "Suno Notícias",
                "api": "https://www.suno.com.br/noticias/wp-json/wp/v2/posts",
                "termos": ["Ibovespa", "bolsa brasileira"],
            },
        ],
    },

    # --- Sentimento ---
    "finbert": {
        "modelo": "otaviopierin/finbert-pt-br-ft",
        # Depois de rodar finetune_finbert.py, troque pela versão ajustada:
        # "modelo": "modelos_finbert/finbert-pt-br-ft",
        "batch": 32,
        "max_tokens": 128,
        "usar_resumo": False,          # True = título + resumo; False = só o título
    },
}