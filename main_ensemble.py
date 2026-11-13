"""
Previsão direcional com dois tipos de fusão, avaliados em walk-forward.

LATE FUSION (padrão): cada ramo é treinado sozinho e só as probabilidades são combinadas
    Preço   -> [Modelo 1: LSTM | Bi-LSTM | CNN | Transformer | MLP | SVM | regressão] -> P_A --\
                                                                                             >-- fusão -> P
    Notícias -> [Modelo 2: FinBERT-PT-BR + calibração]                                -> P_B --/

EARLY FUSION: preço e sentimento são concatenados em cada dia da janela, e um único
modelo sequencial (ex.: Transformer) aprende com os dois ao mesmo tempo
    [preço_t | sentimento_t] por dia -> [Transformer] -> P

Uso:
    python main_ensemble.py --temporal lstm
    python main_ensemble.py --temporal transformer --modo early
    python main_ensemble.py --temporal mlp --horizonte 5
    python main_ensemble.py --comparar                  (todas as configurações)
    python main_ensemble.py --comparar --horizonte 5

MÓDULO DE RISCO (--alvo risco): prevê se o próximo pregão terá oscilação acima da mediana
dos últimos 60 pregões (dia de risco alto). Inclui um baseline de persistência sem IA
("amanhã oscila como hoje"), que os modelos precisam superar.
    python main_ensemble.py --alvo risco --temporal transformer --modo early
    python main_ensemble.py --alvo risco --comparar

PREVISÃO SELETIVA (sempre calculada): o modelo só emite sinal nos dias de maior confiança.
O limiar de confiança é escolhido no bloco meta de cada fold e aplicado ao teste.
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

import modelos
from avaliacao import metricas, metricas_seletivas, salvar_matriz_confusao, selecao_por_confianca, tabela
from dados_modelo import carregar, gerar_folds, montar_fold, resumo
from ensemble import LateFusion


def rodar_fold(b, F, nome, modo, fusao, limiar, seed):
    """Treina e prevê um fold. Retorna probabilidades e limiares no teste + previsão de amanhã."""
    i, y = F.idx, b.y
    temporal, tipo = modelos.criar_temporal(nome, seed)

    if modo == "early":
        if tipo != "sequencia":
            raise ValueError(f"Early fusion precisa de um modelo sequencial; '{nome}' é tabular.")
        X = F.X_early
        temporal.fit(X[i["treino"]], y[i["treino"]], X[i["val"]], y[i["val"]])
        p = temporal.predict_proba(X[i["teste"]])
        p_meta = temporal.predict_proba(X[i["meta"]])
        return ({"Early fusion": p}, {"Early fusion": p_meta}, {"Early fusion": 0.5},
                {"Early fusion": float(temporal.predict_proba(X[-1:])[0])}, (temporal, None, None))

    X = F.X_seq if tipo == "sequencia" else F.X_tab
    temporal.fit(X[i["treino"]], y[i["treino"]], X[i["val"]], y[i["val"]])
    texto = modelos.criar_texto(seed).fit(F.X_texto[i["treino"]], y[i["treino"]],
                                          F.X_texto[i["val"]], y[i["val"]])
    pA = {k: temporal.predict_proba(X[i[k]]) for k in ("meta", "teste")}
    pB = {k: texto.predict_proba(F.X_texto[i[k]]) for k in ("meta", "teste")}
    fus = LateFusion(fusao, limiar).fit(pA["meta"], pB["meta"], y[i["meta"]])

    a = float(temporal.predict_proba(X[-1:])[0])
    t = float(texto.predict_proba(F.X_texto[-1:])[0])
    amanha = {"Modelo 1 (temporal)": a, "Modelo 2 (texto)": t,
              "Late fusion": float(fus.predict_proba(np.array([a]), np.array([t]))[0])}
    probs = {"Modelo 1 (temporal)": pA["teste"], "Modelo 2 (texto)": pB["teste"],
             "Late fusion": fus.predict_proba(pA["teste"], pB["teste"])}
    meta = {"Modelo 1 (temporal)": pA["meta"], "Modelo 2 (texto)": pB["meta"],
            "Late fusion": fus.predict_proba(pA["meta"], pB["meta"])}
    limiares = {"Modelo 1 (temporal)": 0.5, "Modelo 2 (texto)": 0.5, "Late fusion": fus.limiar}
    return probs, meta, limiares, amanha, (temporal, texto, fus)


def classe_constante(b, folds):
    """Classe mais comum no treino do primeiro fold: o 'chute' de um preditor sem IA."""
    return int(b.y[folds[0]["treino"]].mean() >= 0.5)


def avaliar_seletivo(b, folds, metas, testes, lims, coberturas):
    """Junta os folds e mede a acurácia nos dias com sinal, para cada cobertura."""
    y = b.y[np.concatenate([f["teste"] for f in folds])]
    p, lim = np.concatenate(testes), np.concatenate(lims)
    saida = {}
    for c in coberturas:
        coberto = np.concatenate([selecao_por_confianca(pm, pt, c) for pm, pt in zip(metas, testes)])
        saida[c] = metricas_seletivas(y, p, lim, coberto, classe_constante(b, folds), passo=b.horizonte)
    return saida


def persistencia(b, folds, coberturas):
    """
    Baseline sem IA para o alvo de risco: 'os próximos dias oscilam como os últimos'.
    Compara a oscilação média dos últimos 5 pregões com a mediana dessa mesma medida nos
    60 anteriores (só passado); razão acima de 1 prevê risco alto. Explora o agrupamento
    de volatilidade, então é um baseline forte: a IA precisa superá-lo para agregar valor.
    """
    recente = b.df["Abs_Retorno"].rolling(max(5, b.horizonte)).mean()
    referencia = recente.rolling(60, min_periods=20).median()
    razao = (recente / referencia).values[:b.n]
    prob = 1 / (1 + np.exp(-3 * np.log(np.clip(razao, 1e-6, None))))
    testes = [prob[f["teste"]] for f in folds]
    metas = [prob[f["meta"]] for f in folds]
    lims = [np.full(len(f["teste"]), 0.5) for f in folds]
    teste = np.concatenate([f["teste"] for f in folds])
    res = [metricas(b.y[teste], np.concatenate(testes), 0.5, b.ret_futuro[teste],
                    passo=b.horizonte, alvo=b.alvo)]
    por_fold = np.array([[((pt >= 0.5) == b.y[f["teste"]]).mean() for pt, f in zip(testes, folds)]])
    sel = {c: [m] for c, m in avaliar_seletivo(b, folds, metas, testes, lims, coberturas).items()}
    return res, por_fold, sel


def executar(b, folds, nome, modo, fusao, limiar, sementes, coberturas=(1.0,), verbose=True):
    """
    Para cada semente, junta as previsões de todos os folds de teste e calcula as métricas
    sobre esse período fora da amostra completo. Também guarda a acurácia de cada fold.
    """
    dados_folds = [montar_fold(b, f) for f in folds]
    teste = np.concatenate([f["teste"] for f in folds])
    res, por_fold, sel = {}, {}, {}
    for s, seed in enumerate(sementes):
        inicio = time.time()
        juntos, lims, metas = {}, {}, {}
        for k, F in enumerate(dados_folds):
            probs, meta, limiares, amanha, objs = rodar_fold(b, F, nome, modo, fusao, limiar, seed)
            for comp, p in probs.items():
                juntos.setdefault(comp, []).append(p)
                metas.setdefault(comp, []).append(meta[comp])
                lims.setdefault(comp, []).append(np.full(len(p), limiares[comp]))
                acc = ((p >= limiares[comp]) == b.y[F.idx["teste"]]).mean()
                por_fold.setdefault(comp, np.zeros((len(sementes), len(folds))))[s, k] = acc
        for comp in juntos:
            p, lim = np.concatenate(juntos[comp]), np.concatenate(lims[comp])
            res.setdefault(comp, []).append(
                metricas(b.y[teste], p, lim, b.ret_futuro[teste], passo=b.horizonte, alvo=b.alvo))
            for c, m in avaliar_seletivo(b, folds, metas[comp], juntos[comp], lims[comp], coberturas).items():
                sel.setdefault(comp, {}).setdefault(c, []).append(m)
        if verbose:
            principal = "Early fusion" if modo == "early" else "Late fusion"
            print(f"   semente {seed}: acurácia {res[principal][-1]['acuracia']:.2%} "
                  f"| AUC {res[principal][-1]['auc']:.3f} | {time.time() - inicio:.0f}s")
    ultimo = {"probs": {c: np.concatenate(v) for c, v in juntos.items()},
              "lims": {c: np.concatenate(v) for c, v in lims.items()},
              "amanha": amanha, "objs": objs, "teste": teste,
              # fim do treino do último fold: o app usa para refazer a mesma normalização
              "fim_treino": str(b.datas[folds[-1]["treino"][-1]].date())}
    return res, por_fold, sel, ultimo


def salvar_saidas(b, nome_cfg, ultimo, pasta):
    os.makedirs(pasta, exist_ok=True)
    pre = os.path.join(pasta, nome_cfg)
    t = ultimo["teste"]
    tab = pd.DataFrame({"Data": b.datas[t], "Real": b.y[t].astype(int),
                        "Retorno_Futuro": b.ret_futuro[t]})
    for comp, p in ultimo["probs"].items():
        tab[f"P_{comp}"] = p
        sufixo = {"Modelo 1 (temporal)": "temporal", "Modelo 2 (texto)": "texto"}.get(
            comp, comp.split(" ")[0].lower())
        salvar_matriz_confusao(b.y[t], p, ultimo["lims"][comp], f"{comp} - {nome_cfg}",
                               f"{pre}_matriz_{sufixo}.png")
    tab.to_csv(f"{pre}_previsoes_teste.csv", index=False)

    # modelos do último fold (treinados com o maior histórico), para uso em produção
    temporal, texto, fus = ultimo["objs"]
    extras = {"janela": b.janela, "horizonte": b.horizonte, "alvo": b.alvo,
              "features_preco": b.feat_preco, "fim_treino": ultimo["fim_treino"]}
    temporal.salvar(f"{pre}_modelo1{'.pth' if hasattr(temporal, 'rede') else '.joblib'}", extras)
    if texto is not None:
        texto.salvar(f"{pre}_modelo2.joblib", {**extras, "features": b.feat_texto})
        joblib.dump(fus, f"{pre}_fusao.joblib")


def imprimir_folds(b, folds, por_fold_geral):
    base = np.array([max(b.y[f["teste"]].mean(), 1 - b.y[f["teste"]].mean()) for f in folds])
    linhas = {"baseline do fold": base}
    for nome, matriz in por_fold_geral.items():
        linhas[nome] = matriz.mean(axis=0)
    df = pd.DataFrame(linhas, index=[f"fold {k + 1}" for k in range(len(folds))]).T
    df["acima do baseline"] = [f"{int((df.loc[n].values[:len(folds)] > base).sum())}/{len(folds)}"
                               if n != "baseline do fold" else "" for n in df.index]
    print("\nAcurácia por fold (média entre sementes):")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string())
    return df


def tabela_seletiva(sel_geral, coberturas):
    """Linha = configuração; colunas = acurácia (e p-valor) em cada cobertura."""
    linhas = {}
    for nome, por_cob in sel_geral.items():
        linha = {}
        for c in coberturas:
            lst = por_cob[c]
            acc = np.nanmean([m["acuracia"] for m in lst])
            base = np.nanmean([m["baseline"] for m in lst])
            pv = np.nanmean([m["p_valor"] for m in lst])
            cob = np.mean([m["cobertura_real"] for m in lst])
            linha[f"{c:.0%}"] = f"{acc:.1%} (base {base:.1%}, p={pv:.3f}, cob. {cob:.0%})"
        linhas[nome] = linha
    return pd.DataFrame(linhas).T


def main():
    ap = argparse.ArgumentParser(description="Late/early fusion com avaliação walk-forward.")
    ap.add_argument("--base", default="dados/base_diaria_BVSP.csv")
    ap.add_argument("--alvo", default="direcao", choices=["direcao", "risco"],
                    help="direcao = alta/baixa; risco = oscilação acima do normal")
    ap.add_argument("--temporal", default="lstm", choices=list(modelos.TEMPORAIS))
    ap.add_argument("--modo", default="late", choices=["late", "early"])
    ap.add_argument("--fusao", default="media", choices=["media", "ponderada", "stacking"])
    ap.add_argument("--limiar", default="fixo", choices=["fixo", "meta"])
    ap.add_argument("--horizonte", type=int, default=1, help="prever h pregões à frente")
    ap.add_argument("--janela", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tam-teste", type=int, default=126, help="pregões por fold de teste")
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--coberturas", default="1.0,0.5,0.3,0.2",
                    help="frações de dias com sinal na previsão seletiva")
    ap.add_argument("--comparar", action="store_true")
    ap.add_argument("--saida", default="resultados")
    a = ap.parse_args()
    coberturas = [float(c) for c in a.coberturas.split(",")]

    b = carregar(a.base, a.janela, a.horizonte, a.alvo)
    folds = gerar_folds(b, a.folds, a.tam_teste)
    resumo(b, folds)
    sementes = list(range(42, 42 + a.sementes))
    rotulo_p = "P(risco alto)" if a.alvo == "risco" else "P(alta)"

    if a.comparar:
        configs = [(n, "late") for n in modelos.TEMPORAIS] + [("transformer", "early"), ("lstm", "early")]
    else:
        configs = [(a.temporal, a.modo)]

    geral, folds_geral, sel_geral = {}, {}, {}
    if a.alvo == "risco":
        r, pf, sl = persistencia(b, folds, coberturas)
        chave = "persistência (regra, sem IA)"
        geral[chave], folds_geral[chave], sel_geral[chave] = r, pf, sl
        print(f"\n=== baseline de persistência: acurácia {r[0]['acuracia']:.2%} | AUC {r[0]['auc']:.3f} ===")

    for nome, modo in configs:
        print(f"\n=== {nome} ({modo}{f', fusão {a.fusao}' if modo == 'late' else ''}) ===")
        sem = sementes[:1] if nome in ("svm", "logistica") else sementes   # determinísticos
        res, por_fold, sel, ultimo = executar(b, folds, nome, modo, a.fusao, a.limiar, sem, coberturas)
        for comp, lista in res.items():
            if comp == "Modelo 2 (texto)":
                chave = "texto (sozinho)"
            elif comp == "Modelo 1 (temporal)":
                chave = nome
            elif modo == "late":
                chave = f"{nome} + texto (late)"
            else:
                chave = f"{nome} preço+texto (early)"
            geral[chave], folds_geral[chave], sel_geral[chave] = lista, por_fold[comp], sel[comp]
        if not a.comparar:
            salvar_saidas(b, f"{a.alvo}_{nome}_{modo}_h{a.horizonte}", ultimo, a.saida)
        prox = (b.df.index[-1] + pd.offsets.BDay(a.horizonte)).strftime("%d/%m/%Y")
        print(f"   {rotulo_p} para {prox}: " +
              " | ".join(f"{c} {p:.1%}" for c, p in ultimo["amanha"].items()))

    t = tabela(geral)
    colunas = ["acuracia", "baseline", "ganho_pp", "p_valor", "auc", "f1_macro", "prev_alta_pct",
               "retorno_estrategia_pct", "retorno_buy_hold_pct"]
    if a.alvo == "risco":
        colunas += ["vol_estrategia_pct", "vol_buy_hold_pct"]
    print("\n" + "=" * 110)
    print(f" RESULTADOS FORA DA AMOSTRA ({a.alvo}): {len(folds)} folds x {a.tam_teste} pregões "
          f"(horizonte {a.horizonte}; média ± desvio entre sementes)")
    print("=" * 110)
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print(t[colunas].to_string())
    df_folds = imprimir_folds(b, folds, folds_geral)

    ts = tabela_seletiva(sel_geral, coberturas)
    print("\nPREVISÃO SELETIVA: acurácia só nos dias com sinal (limiar de confiança escolhido no bloco meta)")
    with pd.option_context("display.width", 260, "display.max_colwidth", 60):
        print(ts.to_string())

    os.makedirs(a.saida, exist_ok=True)
    sufixo = f"{a.alvo}_h{a.horizonte}"
    nome_tab = f"comparacao_{sufixo}" if a.comparar else f"{a.temporal}_{a.modo}_{sufixo}"
    t.to_csv(os.path.join(a.saida, f"metricas_{nome_tab}.csv"))
    df_folds.to_csv(os.path.join(a.saida, f"folds_{nome_tab}.csv"))
    ts.to_csv(os.path.join(a.saida, f"seletiva_{nome_tab}.csv"))
    with open(os.path.join(a.saida, f"config_{nome_tab}.json"), "w", encoding="utf-8") as f:
        json.dump(vars(a), f, ensure_ascii=False, indent=2)
    print(f"\n[SALVO] {a.saida}/metricas_, folds_ e seletiva_{nome_tab}.csv")
    if a.alvo == "risco":
        print("Leitura: o modelo só agrega valor se superar a PERSISTÊNCIA (regra sem IA), e não só os 50%.")
    else:
        print("Leitura: um modelo só supera o acaso com ganho_pp > 0, p_valor < 0,05 e acurácia "
              "acima do baseline na maioria dos folds.")


if __name__ == "__main__":
    main()