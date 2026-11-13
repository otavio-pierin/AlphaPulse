"""
Métricas de avaliação para previsão direcional.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def metricas(y, p, limiar=0.5, ret_futuro=None, passo=1, alvo="direcao"):
    """
    passo = horizonte da previsão. Com horizonte > 1, os alvos de dias vizinhos se
    sobrepõem (não são independentes), então o p-valor e o backtest usam só um dia a
    cada 'passo' pregões, sem sobreposição. Sem isso o p-valor sairia otimista demais.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    pred = (p >= limiar).astype(int)
    sub = np.arange(0, len(y), passo)
    acertos_sub = int((pred[sub] == y[sub]).sum())
    baseline = max(y.mean(), 1 - y.mean())
    m = {
        "acuracia": accuracy_score(y, pred),
        "baseline": baseline,
        "ganho_pp": (accuracy_score(y, pred) - baseline) * 100,
        # p-valor: chance de acertar tanto ou mais só chutando a classe majoritária
        "p_valor": binomtest(acertos_sub, len(sub), baseline, alternative="greater").pvalue,
        "auc": roc_auc_score(y, p),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "precisao_alta": precision_score(y, pred, zero_division=0),
        "recall_alta": recall_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "prev_alta_pct": pred.mean() * 100,
    }
    if ret_futuro is not None:
        r = np.asarray(ret_futuro)[sub]
        # direção: comprado nos dias previstos como alta
        # risco: comprado só nos dias previstos como calmos (fora do mercado nos de risco)
        investido = (pred[sub] == 1) if alvo == "direcao" else (pred[sub] == 0)
        estrategia = np.where(investido, r, 0.0)
        m["retorno_estrategia_pct"] = (np.exp(np.sum(estrategia)) - 1) * 100
        m["retorno_buy_hold_pct"] = (np.exp(np.sum(r)) - 1) * 100
        if alvo == "risco":                 # volatilidade anualizada: o objetivo é reduzir risco
            m["vol_estrategia_pct"] = np.std(estrategia) * np.sqrt(252 / passo) * 100
            m["vol_buy_hold_pct"] = np.std(r) * np.sqrt(252 / passo) * 100
    return m


def selecao_por_confianca(p_meta, p_teste, cobertura):
    """
    Previsão seletiva: só emite sinal quando |P - 0,5| é alto. O limiar que dá a cobertura
    desejada é calculado nas previsões do bloco META (nunca no teste) e aplicado ao teste.
    """
    if cobertura >= 1.0:
        return np.ones(len(p_teste), dtype=bool)
    limite = np.quantile(np.abs(np.asarray(p_meta) - 0.5), 1 - cobertura)
    return np.abs(np.asarray(p_teste) - 0.5) >= limite


def metricas_seletivas(y, p, limiar, coberto, classe_constante):
    """
    Acurácia só nos dias com sinal. O baseline é um preditor constante que sempre responde
    a classe mais comum no TREINO (escolhida sem olhar o teste), avaliado nos mesmos dias.
    Comparar com a classe majoritária dos próprios dias selecionados seria usar o teste.
    """
    y, p, coberto = np.asarray(y).astype(int), np.asarray(p), np.asarray(coberto)
    lim = np.broadcast_to(np.asarray(limiar, dtype=float), p.shape)
    if coberto.sum() < 10:
        return {"cobertura_real": coberto.mean(), "n_sinais": int(coberto.sum()), "acuracia": np.nan,
                "baseline": np.nan, "ganho_pp": np.nan, "p_valor": np.nan}
    yc, pc = y[coberto], (p[coberto] >= lim[coberto]).astype(int)
    acc = (pc == yc).mean()
    base = float((yc == classe_constante).mean())
    return {"cobertura_real": coberto.mean(), "n_sinais": int(coberto.sum()), "acuracia": acc,
            "baseline": base, "ganho_pp": (acc - base) * 100,
            "p_valor": binomtest(int((pc == yc).sum()), len(yc), max(base, 0.5),
                                 alternative="greater").pvalue}


def tabela(resultados):
    """resultados: {nome: [dict de métricas por semente]} -> DataFrame média ± desvio."""
    linhas = {}
    for nome, lista in resultados.items():
        df = pd.DataFrame(lista)
        linhas[nome] = {c: (f"{df[c].mean():.3f} ± {df[c].std(ddof=0):.3f}" if len(df) > 1
                            else f"{df[c].mean():.3f}") for c in df.columns}
    return pd.DataFrame(linhas).T


def salvar_matriz_confusao(y, p, limiar, titulo, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    pred = (np.asarray(p) >= np.asarray(limiar)).astype(int)
    cm = confusion_matrix(np.asarray(y).astype(int), pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=130)
    ConfusionMatrixDisplay(cm, display_labels=["Classe 0", "Classe 1"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)