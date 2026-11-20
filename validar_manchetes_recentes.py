"""
Passo 2 da validação em manchetes REAIS: compara os rótulos humanos com o score do modelo.

Entrada: um ou dois arquivos preenchidos (colunas id e rotulo), gerados por amostrar_manchetes.py.
  - Com dois anotadores: calcula a concordância (kappa de Cohen) e usa só as manchetes em que
    os dois concordam como "gabarito". Os desacordos ficam registrados e dizem o quanto a
    própria tarefa é ambígua.
  - Com um anotador: usa os rótulos dele e avisa que não há medida de concordância.

Compara com a classificação do modelo que está em dados/noticias_<ativo>.csv, com a MESMA regra
do app: score >= +0,15 positiva, <= -0,15 negativa, entre os dois neutra.

Saídas (pasta resultados/):
    validacao_manchetes_recentes.csv          resumo das métricas
    validacao_manchetes_recentes_por_ano.csv  acurácia por ano
    validacao_manchetes_recentes_erros.csv    manchetes em que o modelo errou (para a análise crítica)
    validacao_manchetes_recentes_matriz.png   matriz de confusão

Uso:
    python validar_manchetes_recentes.py dados/validacao_manchetes_A.csv dados/validacao_manchetes_B.csv
    python validar_manchetes_recentes.py dados/validacao_manchetes_A.csv        (um anotador só)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from servico_app import caminhos

CLASSES = ["neg", "neu", "pos"]
NOMES = {"neg": "negativa", "neu": "neutra", "pos": "positiva"}
MAPA = {"pos": "pos", "positiva": "pos", "positivo": "pos", "1": "pos", "+": "pos",
        "neg": "neg", "negativa": "neg", "negativo": "neg", "-1": "neg", "-": "neg",
        "neu": "neu", "neutra": "neu", "neutro": "neu", "0": "neu"}
LIMIAR = 0.15           # o mesmo de servico_app.classe_sentimento
REFERENCIA_UNICAMP = os.path.join("resultados", "finetune_finbert_cv.csv")


def ler_rotulos(caminho):
    """Lê um arquivo de rotulagem (aceita ';', ',' ou tab) e normaliza os rótulos."""
    df = None
    for sep in (";", ",", "\t"):
        t = pd.read_csv(caminho, sep=sep, encoding="utf-8-sig", dtype=str)
        if {"id", "rotulo"} <= set(t.columns):
            df = t
            break
    if df is None:
        raise SystemExit(f"{caminho}: preciso das colunas 'id' e 'rotulo'.")
    bruto = df["rotulo"].fillna("").str.strip().str.lower()
    df["rotulo"] = bruto.map(MAPA)
    invalidos = df[(bruto != "") & df["rotulo"].isna()]
    if len(invalidos):
        exemplos = ", ".join(sorted(set(bruto[invalidos.index]))[:5])
        raise SystemExit(f"{caminho}: rótulos inválidos ({exemplos}). Use pos, neg ou neu.")
    return df[["id", "rotulo"]].set_index("id")["rotulo"]


def classe_modelo(score, limiar=LIMIAR):
    return np.where(score >= limiar, "pos", np.where(score <= -limiar, "neg", "neu"))


def ic_bootstrap(acertos, rodadas=2000, seed=42):
    """Intervalo de 95% da acurácia por bootstrap (amostras pequenas pedem cautela)."""
    rng = np.random.default_rng(seed)
    acertos = np.asarray(acertos, dtype=float)
    medias = [rng.choice(acertos, len(acertos)).mean() for _ in range(rodadas)]
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def referencia_unicamp():
    """Acurácia binária do FinBERT ajustado na base da Unicamp (validação cruzada), se existir."""
    if not os.path.exists(REFERENCIA_UNICAMP):
        return None
    t = pd.read_csv(REFERENCIA_UNICAMP, index_col=0)
    if "FinBERT ajustado (fine-tuning)" not in t.index:
        return None
    return t.loc["FinBERT ajustado (fine-tuning)", "acuracia"]


def main():
    ap = argparse.ArgumentParser(description="Valida o FinBERT nas manchetes rotuladas pelo grupo.")
    ap.add_argument("arquivos", nargs="+", help="1 ou 2 arquivos preenchidos")
    ap.add_argument("--saida", default="resultados")
    a = ap.parse_args()
    if len(a.arquivos) > 2:
        raise SystemExit("Passe no máximo 2 arquivos (dois anotadores).")

    rotulos = [ler_rotulos(c) for c in a.arquivos]
    r = pd.concat(rotulos, axis=1, join="inner")
    r.columns = [f"anot{i + 1}" for i in range(len(rotulos))]
    n_total = len(r)
    r = r.dropna()                                   # só manchetes rotuladas por todos
    print(f"[rótulos] {len(r)} de {n_total} manchetes rotuladas por todos os anotadores")
    if r.empty:
        raise SystemExit("Nenhuma manchete rotulada. Preencha a coluna 'rotulo'.")

    resumo = {"manchetes_rotuladas": len(r)}
    if len(rotulos) == 2:
        concorda = r["anot1"] == r["anot2"]
        kappa = cohen_kappa_score(r["anot1"], r["anot2"], labels=CLASSES)
        resumo.update({"concordancia_anotadores": round(float(concorda.mean()), 4),
                       "kappa_cohen": round(float(kappa), 4),
                       "descartadas_por_desacordo": int((~concorda).sum())})
        print(f"[anotadores] concordância {concorda.mean():.1%} | kappa de Cohen {kappa:.2f} "
              f"| {int((~concorda).sum())} desacordos descartados")
        gab = r.loc[concorda, "anot1"].rename("humano").to_frame()
    else:
        print("[aviso] só um anotador: sem medida de concordância; o gabarito é a opinião de uma pessoa.")
        gab = r["anot1"].rename("humano").to_frame()

    noticias = pd.read_csv(caminhos()["noticias"])
    noticias["data_efetiva"] = pd.to_datetime(noticias["data_efetiva"])
    m = gab.join(noticias.set_index("id")[["titulo", "data_efetiva", "score", "modelo"]], how="left")
    sem_score = m["score"].isna()
    if sem_score.any():
        print(f"[aviso] {int(sem_score.sum())} manchetes não estão em {caminhos()['noticias']} "
              "(a base foi regenerada?) e foram ignoradas.")
        m = m[~sem_score]
    if m.empty:
        raise SystemExit("Nenhuma manchete rotulada foi encontrada na base de notícias.")
    print(f"[modelo] {', '.join(sorted(m['modelo'].dropna().astype(str).unique()))}")

    m["modelo_prev"] = classe_modelo(m["score"].values)
    m["acertou"] = m["modelo_prev"] == m["humano"]
    n = len(m)
    dist = m["humano"].value_counts().reindex(CLASSES, fill_value=0)
    print(f"[gabarito] {n} manchetes: " + ", ".join(f"{NOMES[c]} {dist[c]}" for c in CLASSES))

    # --- 3 classes (a tarefa completa do app) ---
    acc3 = accuracy_score(m["humano"], m["modelo_prev"])
    f1 = f1_score(m["humano"], m["modelo_prev"], labels=CLASSES, average="macro", zero_division=0)
    base3 = dist.max() / n
    lo3, hi3 = ic_bootstrap(m["acertou"])
    resumo.update({"n_gabarito": n, "acuracia_3_classes": round(acc3, 4),
                   "ic95_3_classes": f"{lo3:.3f}-{hi3:.3f}", "f1_macro_3_classes": round(f1, 4),
                   "baseline_classe_majoritaria": round(base3, 4)})

    # --- binária pos x neg (comparável aos ~88% da Unicamp, que só tem essas classes) ---
    b = m[m["humano"].isin(["pos", "neg"])]
    if len(b) >= 10:
        prev_b = np.where(b["score"] > 0, "pos", "neg")
        acc2 = float((prev_b == b["humano"]).mean())
        base2 = b["humano"].value_counts(normalize=True).max()
        lo2, hi2 = ic_bootstrap(prev_b == b["humano"])
        resumo.update({"n_pos_neg": len(b), "acuracia_pos_x_neg": round(acc2, 4),
                       "ic95_pos_x_neg": f"{lo2:.3f}-{hi2:.3f}", "baseline_pos_x_neg": round(base2, 4)})
    else:
        acc2 = None
        print("[aviso] poucas manchetes pos/neg (<10): a métrica binária foi omitida.")

    # --- por ano ---
    por_ano = (m.assign(ano=m["data_efetiva"].dt.year).groupby("ano")["acertou"]
                .agg(manchetes="size", acuracia_3_classes="mean").round(3))

    # --- erros, para a análise crítica ---
    erros = m[~m["acertou"]].reset_index()[["id", "data_efetiva", "titulo", "humano", "modelo_prev", "score"]]
    erros = erros.sort_values("score", key=lambda s: s.abs(), ascending=False)   # erros "confiantes" primeiro

    # --- matriz de confusão ---
    cm = confusion_matrix(m["humano"], m["modelo_prev"], labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), [NOMES[c] for c in CLASSES])
    ax.set_yticks(range(3), [NOMES[c] for c in CLASSES])
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Anotadores")
    ax.set_title(f"Manchetes reais (n={n})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()

    os.makedirs(a.saida, exist_ok=True)
    pasta = a.saida
    pd.Series(resumo, name="valor").to_csv(os.path.join(pasta, "validacao_manchetes_recentes.csv"),
                                           index_label="metrica")
    por_ano.to_csv(os.path.join(pasta, "validacao_manchetes_recentes_por_ano.csv"))
    erros.to_csv(os.path.join(pasta, "validacao_manchetes_recentes_erros.csv"), index=False)
    fig.savefig(os.path.join(pasta, "validacao_manchetes_recentes_matriz.png"), dpi=150)

    # --- relatório ---
    print("\n================ RESULTADO ================")
    print(f"3 classes (positiva/neutra/negativa): acurácia {acc3:.1%} (IC95% {lo3:.1%}-{hi3:.1%}), "
          f"F1 macro {f1:.3f}, baseline {base3:.1%}")
    if acc2 is not None:
        ref = referencia_unicamp()
        print(f"Só positiva x negativa ({len(b)}): acurácia {acc2:.1%} (IC95% {lo2:.1%}-{hi2:.1%}), "
              f"baseline {base2:.1%}" + (f" | Unicamp (validação cruzada): {ref}" if ref else ""))
    print("\nPor ano:\n" + por_ano.to_string())
    print(f"\nErros: {len(erros)} (lista em {os.path.join(pasta, 'validacao_manchetes_recentes_erros.csv')})")
    for _, e in erros.head(5).iterrows():
        print(f"  humano={e['humano']} modelo={e['modelo_prev']} ({e['score']:+.2f}) | {e['titulo'][:90]}")
    print("\nArquivos salvos em", pasta)


if __name__ == "__main__":
    main()
