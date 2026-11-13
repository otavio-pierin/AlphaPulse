"""
Fine-tuning do FinBERT-PT-BR com as notícias rotuladas da Unicamp (Labeled.zip).

Por que: a validação mostrou que o modelo original é pessimista (recall de 95,7% nas
negativas, mas só 63,3% nas positivas) e erra padrões típicos do mercado brasileiro,
como "dólar cai" (positivo para a bolsa). O fine-tuning ensina esses padrões.

Avaliação sem vazamento:
  - as notícias são divididas em treino (70%), validação (10%) e TESTE (20%), estratificadas
    por classe e fonte;
  - o teste nunca é usado no treino nem na escolha de época ou limiar;
  - o FinBERT original e o ajustado são comparados nas MESMAS notícias de teste.
  - --cv 5 repete tudo em 5 divisões diferentes e reporta média ± desvio (mais robusto).

Uso:
    python finetune_finbert.py --arquivo Labeled.zip              (treina, avalia e salva)
    python finetune_finbert.py --arquivo Labeled.zip --cv 5       (só avaliação, 5 divisões)

Depois, em config.py:  "modelo": "modelos_finbert/finbert-pt-br-ft"
e rode gerar_base.py: as manchetes são pontuadas de novo com o modelo ajustado.

ATENÇÃO: depois do fine-tuning, não use avaliar_finbert.py na base inteira para medir o
modelo ajustado; ele viu 70% dessas notícias. O número válido é o do conjunto de teste.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from avaliar_finbert import carregar_rotulados

MODELO_BASE = "lucas-leme/FinBERT-PT-BR"
NOMES = {0: "negativa", 1: "positiva", 2: "neutra"}       # rótulos da base


# -----------------------------------------------------------------------------------
# Dados
# -----------------------------------------------------------------------------------
def estratos(df):
    return df["fonte"].astype(str) + "|" + df["rotulo"].astype(str)


def dividir(df, seed=42, prop_teste=0.2, prop_val=0.1):
    """Treino / validação / teste estratificados por fonte e classe."""
    idx_resto, idx_teste = train_test_split(df.index, test_size=prop_teste, random_state=seed,
                                            stratify=estratos(df))
    resto = df.loc[idx_resto]
    idx_tr, idx_va = train_test_split(resto.index, test_size=prop_val / (1 - prop_teste),
                                      random_state=seed, stratify=estratos(resto))
    return np.array(idx_tr), np.array(idx_va), np.array(idx_teste)


# -----------------------------------------------------------------------------------
# Modelo
# -----------------------------------------------------------------------------------
def fixar_sementes(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def carregar_modelo(nome):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nome)
    try:
        modelo = AutoModelForSequenceClassification.from_pretrained(nome, use_safetensors=True)
    except (OSError, ValueError):
        modelo = AutoModelForSequenceClassification.from_pretrained(nome)
    return tok, modelo


def mapa_rotulos(modelo):
    """Rótulo da base (0 neg, 1 pos, 2 neutra) -> índice da saída do modelo."""
    nomes = {i: str(r).lower() for i, r in modelo.config.id2label.items()}
    achar = lambda chave: next(i for i, r in nomes.items() if chave in r)
    return {0: achar("neg"), 1: achar("pos"), 2: achar("neu")}


def prever(modelo, tok, textos, max_tokens=96, batch=64):
    """Probabilidades (n, 3) já reordenadas para [negativa, positiva, neutra]."""
    import torch
    disp = next(modelo.parameters()).device
    mapa = mapa_rotulos(modelo)
    ordem = [mapa[0], mapa[1], mapa[2]]
    modelo.eval()
    saida = []
    with torch.no_grad():
        for i in range(0, len(textos), batch):
            enc = tok(list(textos[i:i + batch]), padding=True, truncation=True,
                      max_length=max_tokens, return_tensors="pt").to(disp)
            with torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
                logits = modelo(**enc).logits
            saida.append(torch.softmax(logits.float(), dim=-1)[:, ordem].cpu().numpy())
    return np.vstack(saida)


def treinar(df_tr, df_va, args, verbose=True):
    """Fine-tuning com AdamW, warmup linear, precisão mista e early stopping na validação."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    fixar_sementes(args.seed)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, modelo = carregar_modelo(args.modelo)
    modelo.to(disp)
    mapa = mapa_rotulos(modelo)

    textos = df_tr["texto"].tolist()
    alvos = torch.tensor([mapa[r] for r in df_tr["rotulo"]], dtype=torch.long)

    # Pesos de classe: a neutra é rara na base (só as notícias com 2 anotadores)
    contagem = np.bincount(alvos.numpy(), minlength=modelo.config.num_labels).astype(float)
    pesos = np.where(contagem > 0, contagem.sum() / (len(contagem) * np.maximum(contagem, 1)), 0.0)
    criterio = torch.nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32, device=disp))

    sem_decay = ("bias", "LayerNorm.weight")
    grupos = [
        {"params": [p for n, p in modelo.named_parameters() if not any(s in n for s in sem_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in modelo.named_parameters() if any(s in n for s in sem_decay)],
         "weight_decay": 0.0},
    ]
    otim = torch.optim.AdamW(grupos, lr=args.lr)
    passos = args.epocas * int(np.ceil(len(textos) / args.batch))
    agenda = get_linear_schedule_with_warmup(otim, int(0.1 * passos), passos)
    escala = torch.amp.GradScaler("cuda", enabled=disp.type == "cuda")

    melhor, melhor_estado, sem_melhora = -1.0, None, 0
    for ep in range(args.epocas):
        modelo.train()
        ordem = np.random.permutation(len(textos))
        soma, inicio = 0.0, time.time()
        for i in range(0, len(ordem), args.batch):
            sel = ordem[i:i + args.batch]
            enc = tok([textos[j] for j in sel], padding=True, truncation=True,
                      max_length=args.max_tokens, return_tensors="pt").to(disp)
            with torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
                perda = criterio(modelo(**enc).logits.float(), alvos[sel].to(disp))
            otim.zero_grad()
            escala.scale(perda).backward()
            escala.unscale_(otim)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escala.step(otim)
            escala.update()
            agenda.step()
            soma += perda.item() * len(sel)

        m = avaliar(df_va, prever(modelo, tok, df_va["texto"].tolist(), args.max_tokens))
        if verbose:
            print(f"   época {ep + 1}/{args.epocas} | perda {soma / len(textos):.4f} | "
                  f"val acurácia {m['acuracia']:.2%} | val F1 {m['f1_macro']:.3f} | {time.time() - inicio:.0f}s")
        if m["f1_macro"] > melhor + 1e-4:
            melhor, sem_melhora = m["f1_macro"], 0
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                if verbose:
                    print(f"   early stopping (melhor F1 de validação {melhor:.3f})")
                break
    modelo.load_state_dict(melhor_estado)
    return tok, modelo


# -----------------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------------
def avaliar(df, probs, limiar=None):
    """
    Binária (positiva x negativa) nas notícias sem rótulo neutro.
    score = P(pos) - P(neg); previsão positiva se score > limiar (padrão: P(pos) > P(neg)).
    """
    binaria = (df["rotulo"] != 2).values
    y = df["rotulo"].values[binaria]
    score = probs[binaria, 1] - probs[binaria, 0]
    pred = (score > (0.0 if limiar is None else limiar)).astype(int)
    m = {"n": int(binaria.sum()), "acuracia": accuracy_score(y, pred),
         "f1_macro": f1_score(y, pred, average="macro"), "auc": roc_auc_score(y, score),
         "recall_neg": (pred[y == 0] == 0).mean(), "recall_pos": (pred[y == 1] == 1).mean(),
         "baseline": max(y.mean(), 1 - y.mean())}
    if (df["rotulo"] == 2).any():                          # três classes
        y3 = df["rotulo"].values
        m["acuracia_3_classes"] = accuracy_score(y3, probs.argmax(axis=1))
    return m


def limiar_otimo(df, probs):
    """Limiar do score que maximiza a acurácia (escolhido em treino/validação, nunca no teste)."""
    binaria = (df["rotulo"] != 2).values
    y, s = df["rotulo"].values[binaria], probs[binaria, 1] - probs[binaria, 0]
    candidatos = np.linspace(-0.95, 0.95, 191)
    return float(max(candidatos, key=lambda l: accuracy_score(y, s > l)))


def comparar(df, idx_tr, idx_va, idx_te, probs_orig, args, verbose=True):
    """Original (padrão e com limiar calibrado) x ajustado, no mesmo teste."""
    te = df.loc[idx_te]
    lim_orig = limiar_otimo(df.loc[np.concatenate([idx_tr, idx_va])], probs_orig[np.concatenate([idx_tr, idx_va])])
    res = {
        "FinBERT original": avaliar(te, probs_orig[idx_te]),
        "FinBERT original + limiar calibrado": avaliar(te, probs_orig[idx_te], lim_orig),
    }
    tok, modelo = treinar(df.loc[idx_tr], df.loc[idx_va], args, verbose)
    probs_va = prever(modelo, tok, df.loc[idx_va, "texto"].tolist(), args.max_tokens)
    probs_te = prever(modelo, tok, te["texto"].tolist(), args.max_tokens)
    lim_ft = limiar_otimo(df.loc[idx_va], probs_va)
    res["FinBERT ajustado (fine-tuning)"] = avaliar(te, probs_te)
    res["FinBERT ajustado + limiar calibrado"] = avaliar(te, probs_te, lim_ft)
    return res, tok, modelo, probs_te, lim_ft


def imprimir(res, titulo):
    cols = ["n", "acuracia", "baseline", "f1_macro", "auc", "recall_neg", "recall_pos", "acuracia_3_classes"]
    t = pd.DataFrame(res).T.reindex(columns=cols)
    print("\n" + "=" * 100)
    print(f" {titulo}")
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(t.to_string())
    return t


# -----------------------------------------------------------------------------------
# Execução
# -----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tuning do FinBERT-PT-BR com avaliação sem vazamento.")
    ap.add_argument("--arquivo", default="Labeled.zip")
    ap.add_argument("--modelo", default=MODELO_BASE)
    ap.add_argument("--saida", default="modelos_finbert/finbert-pt-br-ft")
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--paciencia", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16, help="reduza para 8 se faltar memória na GPU")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=0, help="k > 1: avaliação em k divisões (não salva modelo)")
    a = ap.parse_args()

    df = carregar_rotulados(a.arquivo).reset_index(drop=True)

    # Probabilidades do modelo original em toda a base (uma vez só, sem treino)
    tok0, orig = carregar_modelo(a.modelo)
    import torch
    orig.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[original] Pontuando {len(df)} notícias com {a.modelo}...")
    probs_orig = prever(orig, tok0, df["texto"].tolist(), a.max_tokens)
    del orig
    torch.cuda.empty_cache()

    if a.cv > 1:
        todos = {}
        skf = StratifiedKFold(a.cv, shuffle=True, random_state=a.seed)
        for k, (idx_resto, idx_te) in enumerate(skf.split(df, estratos(df)), 1):
            resto = df.iloc[idx_resto]
            idx_tr, idx_va = train_test_split(resto.index, test_size=0.125, random_state=a.seed,
                                              stratify=estratos(resto))
            print(f"\n--- divisão {k}/{a.cv}: treino {len(idx_tr)} | validação {len(idx_va)} | teste {len(idx_te)}")
            res, *descartar = comparar(df, np.array(idx_tr), np.array(idx_va), df.index[idx_te].values,
                                       probs_orig, a)
            for nome, m in res.items():
                todos.setdefault(nome, []).append(m)
            del descartar                                  # libera o modelo da GPU antes da próxima divisão
            torch.cuda.empty_cache()
        resumo = {nome: {c: f"{np.mean([m[c] for m in lst]):.3f} ± {np.std([m[c] for m in lst]):.3f}"
                         for c in lst[0]} for nome, lst in todos.items()}
        t = imprimir(resumo, f"Validação cruzada em {a.cv} divisões (média ± desvio no teste de cada divisão)")
        os.makedirs("resultados", exist_ok=True)
        t.to_csv("resultados/finetune_finbert_cv.csv")
        print("\n[SALVO] resultados/finetune_finbert_cv.csv")
        return

    idx_tr, idx_va, idx_te = dividir(df, a.seed)
    print(f"[divisão] treino {len(idx_tr)} | validação {len(idx_va)} | teste {len(idx_te)} (nunca usado no treino)")
    res, tok, modelo, probs_te, lim_ft = comparar(df, idx_tr, idx_va, idx_te, probs_orig, a)
    t = imprimir(res, f"Conjunto de teste ({len(idx_te)} notícias nunca vistas no treino)")

    # Matriz de confusão do ajustado no teste
    te = df.loc[idx_te]
    b = (te["rotulo"] != 2).values
    pred = (probs_te[b, 1] - probs_te[b, 0] > 0).astype(int)
    print("\nMatriz de confusão do modelo ajustado (teste, positivas x negativas):")
    print(pd.DataFrame(confusion_matrix(te["rotulo"].values[b], pred, labels=[0, 1]),
                       index=["real neg", "real pos"], columns=["prev neg", "prev pos"]).to_string())

    # Salvamento: modelo em safetensors + tokenizer + métricas + quais notícias são de teste
    os.makedirs(a.saida, exist_ok=True)
    modelo.save_pretrained(a.saida, safe_serialization=True)
    tok.save_pretrained(a.saida)
    with open(os.path.join(a.saida, "metricas_teste.json"), "w", encoding="utf-8") as f:
        json.dump({"modelo_base": a.modelo, "limiar_score_calibrado": lim_ft,
                   "hiperparametros": {k: v for k, v in vars(a).items() if k not in ("arquivo", "saida")},
                   "teste": {k: {c: float(x) for c, x in v.items()} for k, v in res.items()}},
                  f, ensure_ascii=False, indent=2)
    df.loc[idx_te, ["texto", "rotulo", "fonte"]].to_csv(os.path.join(a.saida, "conjunto_teste.csv"), index=False)
    os.makedirs("resultados", exist_ok=True)
    t.to_csv("resultados/finetune_finbert_teste.csv")

    print(f"\n[SALVO] modelo ajustado em {a.saida}/ (+ metricas_teste.json e conjunto_teste.csv)")
    print(f'Para usar no pipeline, em config.py: "modelo": "{a.saida}" e rode gerar_base.py.')


if __name__ == "__main__":
    main()