"""
Lógica da aplicação (sem interface), para poder ser testada separadamente do Streamlit.

O app usa apenas artefatos gerados pelo pipeline:
  dados/base_diaria_<ativo>.csv, dados/noticias_<ativo>.csv   (gerar_base.py)
  modelos_finbert/finbert-pt-br-ft/                           (finetune_finbert.py)
  resultados/*.csv e modelos salvos                           (main_ensemble.py, finetune_finbert.py)
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from config import CONFIG

# Configurações exibidas como sinais experimentais (as melhores de cada alvo no walk-forward)
SINAIS = {
    "direcao": {"temporal": "transformer", "modo": "early", "chave": "transformer preço+texto (early)",
                "titulo": "Direção do próximo pregão", "classe1": "Alta", "classe0": "Baixa",
                "comando": "python main_ensemble.py --temporal transformer --modo early"},
    "risco": {"temporal": "lstm", "modo": "late", "chave": "lstm",
              "titulo": "Risco do próximo pregão", "classe1": "Risco alto", "classe0": "Risco normal",
              "comando": "python main_ensemble.py --alvo risco --temporal lstm"},
}


def nome_ativo(ticker=None):
    t = ticker or CONFIG["ticker"]
    return t.replace("^", "").replace(".SA", "").replace(".", "_")


def caminhos():
    a = nome_ativo()
    pasta = CONFIG["pasta_dados"]
    return {"base": os.path.join(pasta, f"base_diaria_{a}.csv"),
            "noticias": os.path.join(pasta, f"noticias_{a}.csv")}


# -----------------------------------------------------------------------------------
# Dados
# -----------------------------------------------------------------------------------
def ler_base():
    c = caminhos()["base"]
    if not os.path.exists(c):
        return None
    return pd.read_csv(c, index_col="Data", parse_dates=True)


def ler_noticias():
    c = caminhos()["noticias"]
    if not os.path.exists(c):
        return None
    n = pd.read_csv(c)
    n["data_efetiva"] = pd.to_datetime(n["data_efetiva"])
    n["datahora_utc"] = pd.to_datetime(n["datahora_utc"], utc=True, format="ISO8601")
    n["datahora_brasilia"] = n["datahora_utc"].dt.tz_convert(CONFIG["fuso"])
    return n


def classe_sentimento(score, limite=0.15):
    return "Positiva" if score >= limite else "Negativa" if score <= -limite else "Neutra"


def noticias_dos_pregoes(noticias, datas_pregao, n_pregoes=3):
    """Notícias que entraram no sentimento dos últimos n pregões e as posteriores ao último."""
    datas = pd.DatetimeIndex(datas_pregao)
    ultimo = datas[-1]
    inicio = datas[-n_pregoes - 1] if len(datas) > n_pregoes else datas[0]
    usadas = noticias[(noticias["data_efetiva"] > inicio) & (noticias["data_efetiva"] <= ultimo)]
    posteriores = noticias[noticias["data_efetiva"] > ultimo]
    return usadas, posteriores


# -----------------------------------------------------------------------------------
# Métricas já calculadas pelo pipeline
# -----------------------------------------------------------------------------------
def valor(texto):
    """'0.529 ± 0.004' -> 0.529 ; '0.519' -> 0.519 ; ausente -> nan."""
    m = re.match(r"\s*(-?[\d.]+)", str(texto))
    return float(m.group(1)) if m else np.nan


def _ler_primeiro(*nomes):
    for n in nomes:
        c = os.path.join("resultados", n)
        if os.path.exists(c):
            return pd.read_csv(c, index_col=0), c
    return None, None


def metricas_salvas():
    """Tabelas de validação geradas pelos scripts (cada uma pode não existir)."""
    t = {}
    t["finbert_cv"], _ = _ler_primeiro("finetune_finbert_cv.csv")
    t["finbert_teste"], _ = _ler_primeiro("finetune_finbert_teste.csv")
    t["direcao"], _ = _ler_primeiro("metricas_comparacao_direcao_h1.csv", "metricas_comparacao_h1.csv")
    t["direcao_folds"], _ = _ler_primeiro("folds_comparacao_direcao_h1.csv", "folds_comparacao_h1.csv")
    t["risco"], _ = _ler_primeiro("metricas_comparacao_risco_h1.csv")
    t["risco_folds"], _ = _ler_primeiro("folds_comparacao_risco_h1.csv")
    t["risco_seletiva"], _ = _ler_primeiro("seletiva_comparacao_risco_h1.csv")
    t["risco_h5"], _ = _ler_primeiro("metricas_comparacao_risco_h5.csv")
    return t


def resumo_finbert(tabelas):
    """Acurácia do FinBERT original e ajustado (validação cruzada, se houver)."""
    cv = tabelas.get("finbert_cv")
    if cv is None or "FinBERT ajustado (fine-tuning)" not in cv.index:
        return None
    aj, orig = cv.loc["FinBERT ajustado (fine-tuning)"], cv.loc["FinBERT original"]
    return {"acuracia": aj["acuracia"], "auc": aj["auc"], "baseline": valor(aj["baseline"]),
            "acuracia_original": orig["acuracia"], "acc_num": valor(aj["acuracia"]),
            "acc_orig_num": valor(orig["acuracia"])}


def desempenho_sinal(tabelas, alvo):
    """Métricas fora da amostra da configuração exibida para o alvo."""
    t = tabelas.get(alvo)
    chave = SINAIS[alvo]["chave"]
    if t is None or chave not in t.index:
        return None
    linha = t.loc[chave]
    folds = tabelas.get(f"{alvo}_folds")
    acima = None
    if folds is not None and chave in folds.index and "acima do baseline" in folds.columns:
        acima = folds.loc[chave, "acima do baseline"]
    return {"acuracia": valor(linha["acuracia"]), "baseline": valor(linha["baseline"]),
            "p_valor": valor(linha["p_valor"]), "auc": valor(linha["auc"]), "folds_acima": acima,
            "persistencia": (valor(t.loc["persistência (regra, sem IA)", "acuracia"])
                             if "persistência (regra, sem IA)" in t.index else None)}


# -----------------------------------------------------------------------------------
# Sinais experimentais: carrega o modelo salvo e prevê o próximo pregão
# -----------------------------------------------------------------------------------
def localizar_modelo(alvo, horizonte=1):
    s = SINAIS[alvo]
    for ext in (".pth", ".joblib"):
        c = os.path.join("resultados", f"{alvo}_{s['temporal']}_{s['modo']}_h{horizonte}_modelo1{ext}")
        if os.path.exists(c):
            return c
    return None


def prever_sinal(alvo):
    """
    Refaz a preparação dos dados exatamente como no treino (mesma janela, horizonte e
    normalização ajustada até a data de fim do treino) e calcula a probabilidade do
    próximo pregão com o modelo salvo pelo main_ensemble.py.
    """
    from dados_modelo import carregar, montar_fold

    caminho = localizar_modelo(alvo)
    if caminho is None:
        return None
    s = SINAIS[alvo]
    if caminho.endswith(".pth"):
        import torch
        ck = torch.load(caminho, weights_only=True, map_location="cpu")
    else:
        import joblib
        ck = joblib.load(caminho)

    b = carregar(caminhos()["base"], ck.get("janela", 30), ck.get("horizonte", 1), alvo)
    datas = b.df.index
    if "fim_treino" in ck:
        fim = pd.Timestamp(ck["fim_treino"])
    else:   # modelos salvos antes desta versão: aproxima pelo período até 80% da série
        fim = datas[int(0.8 * b.n)]
    pos = np.arange(b.janela - 1, b.n)
    treino = pos[datas[pos] <= fim]
    F = montar_fold(b, {"treino": treino})

    if caminho.endswith(".pth"):
        import modelos
        from modelos.treino_torch import dispositivo
        modelo, tipo = modelos.criar_temporal(s["temporal"], ck.get("seed", 42))
        X = F.X_early if s["modo"] == "early" else (F.X_seq if tipo == "sequencia" else F.X_tab)
        modelo.rede = modelo.construtor(X.shape, **modelo.hp.get("arquitetura", {}))
        modelo.rede.load_state_dict(ck["state_dict"])
        modelo.rede.to(dispositivo()).eval()
        prob = float(modelo.predict_proba(X[-1:])[0])
    else:
        X = F.X_tab
        prob = float(ck["modelo"].predict_proba(X[-1:])[:, 1][0])

    return {"prob": prob, "data_ref": datas[-1], "fim_treino": fim, "arquivo": caminho,
            "alvo": alvo, "proximo": datas[-1] + pd.offsets.BDay(1)}


# -----------------------------------------------------------------------------------
# FinBERT para a análise de manchetes digitadas
# -----------------------------------------------------------------------------------
def info_modelo_sentimento():
    nome = CONFIG["finbert"]["modelo"]
    ajustado = os.path.isdir(nome)
    metricas = None
    arq = os.path.join(nome, "metricas_teste.json") if ajustado else None
    if arq and os.path.exists(arq):
        with open(arq, encoding="utf-8") as f:
            metricas = json.load(f)
    return {"nome": nome, "ajustado": ajustado, "metricas_teste": metricas}


def analisar_manchete(finbert, texto):
    p = finbert.probabilidades([texto])[0]          # [positivo, negativo, neutro]
    score = float(p[0] - p[1])
    return {"positivo": float(p[0]), "negativo": float(p[1]), "neutro": float(p[2]),
            "score": score, "classe": ["Positiva", "Negativa", "Neutra"][int(np.argmax(p))]}


def arquivos_de_resultado():
    return sorted(glob.glob(os.path.join("resultados", "*.csv")))