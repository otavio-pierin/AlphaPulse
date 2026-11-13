"""
Valida o FinBERT com notícias financeiras brasileiras rotuladas.

Base: Labeled.zip de Carosia, Silva e Coelho (2022), Repositório de Dados de Pesquisa
da Unicamp, doi:10.25824/REDU/GFJHFK (licença CC BY-NC 4.0 - cite no trabalho).

Formatos reconhecidos automaticamente (inclusive zips dentro do zip):
  1. 'rótulo|manchete', uma por linha (1 = positiva, 0 = negativa)
     -> Noticias_Bovespa/Itau/Petrobras/Vale/Pos/Neg.txt
  2. notícia completa com duas anotações no cabeçalho ('1¬ 1¬'), valores -1/0/1
     -> News_Bradesco.zip. Interpretamos como dois anotadores e usamos só as notícias
        em que eles concordam; aqui existe a classe neutra (0)
  3. planilhas (.csv, .xlsx, .json) com colunas de texto e rótulo
  4. arquivos de texto em pastas positivo/negativo

Uso:
    python avaliar_finbert.py --arquivo Labeled.zip
    python avaliar_finbert.py --arquivo Labeled.zip --modelo ProsusAI/finbert

Como a base principal tem 2 classes e o FinBERT tem 3, a previsão binária é positiva
quando P(positivo) > P(negativo).
"""

import argparse
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

POSITIVOS = {"positive", "positivo", "pos", "1", "1.0", "p"}
NEGATIVOS = {"negative", "negativo", "neg", "0", "0.0", "-1", "-1.0", "n"}
LINHA_PIPE = re.compile(r"^\s*(-?1|0)\s*\|(.+)$")
CABECALHO_DUPLO = re.compile(r"^\s*(-?1|0)\s*¬\s*(-?1|0)\s*¬")
NOMES = {0: "negativa", 1: "positiva", 2: "neutra"}


# -----------------------------------------------------------------------------------
# Leitura
# -----------------------------------------------------------------------------------
def _arquivos(caminho):
    """Todos os arquivos de um zip (inclusive zips internos), de uma pasta ou um arquivo."""
    arquivos = {}

    def do_zip(dados, prefixo=""):
        with zipfile.ZipFile(io.BytesIO(dados)) as z:
            for nome in z.namelist():
                if nome.endswith("/"):
                    continue
                conteudo = z.read(nome)
                if nome.lower().endswith(".zip"):
                    do_zip(conteudo, prefixo + nome + "/")
                else:
                    arquivos[prefixo + nome] = conteudo

    if os.path.isdir(caminho):
        for raiz, _, nomes in os.walk(caminho):
            for n in nomes:
                p = os.path.join(raiz, n)
                with open(p, "rb") as f:
                    arquivos[os.path.relpath(p, caminho)] = f.read()
    else:
        with open(caminho, "rb") as f:
            dados = f.read()
        if zipfile.is_zipfile(io.BytesIO(dados)):
            do_zip(dados)
        else:
            arquivos[os.path.basename(caminho)] = dados
    return arquivos


def _texto(conteudo):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return conteudo.decode(enc)
        except UnicodeDecodeError:
            continue
    return conteudo.decode("utf-8", errors="ignore")


def _rotulo(valor):
    v = str(valor).strip().lower()
    return 1 if v in POSITIVOS else 0 if v in NEGATIVOS else None


def _formato_pipe(nome, texto):
    linhas = [l for l in texto.splitlines() if l.strip()]
    casadas = [LINHA_PIPE.match(l) for l in linhas]
    if not linhas or sum(m is not None for m in casadas) / len(linhas) < 0.9:
        return []
    return [{"texto": " ".join(m.group(2).split()), "rotulo": 1 if m.group(1) == "1" else 0,
             "arquivo": nome, "fonte": "manchetes rotuladas"} for m in casadas if m]


def _formato_duas_anotacoes(nome, texto):
    linhas = texto.splitlines()
    m = CABECALHO_DUPLO.match(linhas[0]) if linhas else None
    if not m:
        return []
    a, b = int(m.group(1)), int(m.group(2))
    if a != b:
        return [{"descartada": True}]
    titulo = next((l.strip() for l in linhas[1:] if l.strip() and not l.strip().startswith("¬")), "")
    if not titulo:
        return []
    return [{"texto": " ".join(titulo.split()), "rotulo": {1: 1, -1: 0, 0: 2}[a],
             "arquivo": nome, "fonte": "notícias com 2 anotadores"}]


def _ler_tabela(nome, conteudo):
    ext = os.path.splitext(nome)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(conteudo))
    if ext == ".json":
        return pd.read_json(io.BytesIO(conteudo))
    for sep in (None, ";", "\t", ","):
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(conteudo), sep=sep, engine="python", encoding=enc)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
    return None


def _detectar_colunas(df, col_texto=None, col_rotulo=None):
    if col_rotulo is None:
        for c in df.columns:
            valores = df[c].dropna().map(_rotulo)
            if len(valores) and valores.notna().mean() > 0.95 and valores.nunique() == 2:
                col_rotulo = c
                break
    if col_texto is None:
        textos = [c for c in df.columns if c != col_rotulo and
                  (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
        if textos:
            col_texto = max(textos, key=lambda c: df[c].astype(str).str.len().mean())
    return col_texto, col_rotulo


def carregar_rotulados(caminho, col_texto=None, col_rotulo=None):
    """Rótulos: 1 = positiva, 0 = negativa, 2 = neutra."""
    arquivos = _arquivos(caminho)
    print(f"[base] {len(arquivos)} arquivo(s) encontrados (incluindo zips internos)")

    registros, sem_consenso = [], 0
    for nome, conteudo in arquivos.items():
        if nome.lower().endswith(".txt"):
            texto = _texto(conteudo)
            regs = _formato_pipe(nome, texto) or _formato_duas_anotacoes(nome, texto)
            sem_consenso += sum(1 for r in regs if r.get("descartada"))
            registros += [r for r in regs if not r.get("descartada")]

    if not registros:
        for nome, conteudo in arquivos.items():
            if os.path.splitext(nome)[1].lower() not in (".csv", ".tsv", ".xlsx", ".xls", ".json"):
                continue
            df = _ler_tabela(nome, conteudo)
            if df is None:
                continue
            ct, cr = _detectar_colunas(df, col_texto, col_rotulo)
            if ct in df.columns and cr in df.columns:
                for txt, rot in zip(df[ct].astype(str), df[cr].map(_rotulo)):
                    if rot is not None:
                        registros.append({"texto": txt, "rotulo": rot, "arquivo": nome,
                                          "fonte": "planilha"})

    if not registros:
        for nome, conteudo in arquivos.items():
            pastas = [p.lower() for p in nome.replace("\\", "/").split("/")[:-1]]
            classe = next((1 if "pos" in p else 0 for p in pastas if "pos" in p or "neg" in p), None)
            if classe is not None and _texto(conteudo).strip():
                registros.append({"texto": _texto(conteudo).strip(), "rotulo": classe,
                                  "arquivo": nome, "fonte": "pastas"})

    if not registros:
        raise SystemExit(f"Não consegui identificar textos e rótulos. Arquivos: {list(arquivos)[:10]}. "
                         "Use --coluna-texto e --coluna-rotulo.")

    df = pd.DataFrame(registros)
    # A mesma manchete pode estar em mais de um arquivo: fica uma vez; com rótulos
    # divergentes, é descartada
    df["chave"] = df["texto"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    conflito = df.groupby("chave")["rotulo"].transform("nunique") > 1
    antes = len(df)
    df = df[~conflito].drop_duplicates("chave").drop(columns="chave").reset_index(drop=True)
    print(f"[base] {len(df)} notícias únicas | {antes - len(df)} duplicadas ou conflitantes removidas | "
          f"{sem_consenso} sem consenso entre anotadores ignoradas")
    contagem = df.groupby(["fonte", "rotulo"]).size().unstack(fill_value=0).rename(columns=NOMES)
    print(contagem.to_string())
    return df


# -----------------------------------------------------------------------------------
# Relatórios
# -----------------------------------------------------------------------------------
def relatorio_binario(df, titulo):
    y, pred = df["rotulo"].values, df["previsto"].values
    base = max(y.mean(), 1 - y.mean())
    print(f"\n--- {titulo} ({len(df)} notícias) ---")
    print(f" Acurácia: {accuracy_score(y, pred):.2%}  (baseline da classe majoritária: {base:.2%})")
    print(f" F1 macro: {f1_score(y, pred, average='macro'):.3f} | "
          f"ROC-AUC do score: {roc_auc_score(y, df['score']):.3f}")
    print(f" Classificadas como neutras pelo FinBERT: {(df['classe_finbert'] == 'neutro').mean():.1%}")
    print(pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1]), index=["real neg", "real pos"],
                       columns=["prev neg", "prev pos"]).to_string())
    return {"conjunto": titulo, "n": len(df), "acuracia": accuracy_score(y, pred), "baseline": base,
            "f1_macro": f1_score(y, pred, average="macro"), "auc": roc_auc_score(y, df["score"])}


def relatorio_tres_classes(df):
    mapa = {"negativo": 0, "positivo": 1, "neutro": 2}
    y, p = df["rotulo"].values, df["classe_finbert"].map(mapa).values
    print(f"\n--- Três classes: notícias com consenso de 2 anotadores ({len(df)}) ---")
    print(f" Acurácia: {accuracy_score(y, p):.2%} | F1 macro: {f1_score(y, p, average='macro'):.3f}")
    print(pd.DataFrame(confusion_matrix(y, p, labels=[0, 1, 2]),
                       index=["real neg", "real pos", "real neutra"],
                       columns=["prev neg", "prev pos", "prev neutra"]).to_string())
    return {"conjunto": "3 classes (2 anotadores)", "n": len(df), "acuracia": accuracy_score(y, p),
            "baseline": np.bincount(y).max() / len(y), "f1_macro": f1_score(y, p, average="macro"),
            "auc": np.nan}


def main():
    ap = argparse.ArgumentParser(description="Valida o FinBERT com notícias rotuladas.")
    ap.add_argument("--arquivo", required=True, help="zip, pasta ou planilha com as notícias")
    ap.add_argument("--modelo", default="lucas-leme/FinBERT-PT-BR")
    ap.add_argument("--coluna-texto")
    ap.add_argument("--coluna-rotulo")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--saida", default="resultados/validacao_finbert.csv")
    a = ap.parse_args()

    from sentimento import FinBERT

    df = carregar_rotulados(a.arquivo, a.coluna_texto, a.coluna_rotulo)
    modelo = FinBERT(a.modelo, batch=32, max_tokens=a.max_tokens)
    p = modelo.probabilidades(df["texto"].tolist())
    df["P_positivo"], df["P_negativo"], df["P_neutro"] = p[:, 0], p[:, 1], p[:, 2]
    df["score"] = p[:, 0] - p[:, 1]
    df["previsto"] = (p[:, 0] > p[:, 1]).astype(int)
    df["classe_finbert"] = np.array(["positivo", "negativo", "neutro"])[p.argmax(axis=1)]

    print("\n" + "=" * 70)
    print(f" FinBERT ({a.modelo}) em notícias financeiras brasileiras rotuladas")
    print("=" * 70)
    resumos = []
    binaria = df[df["rotulo"] != 2]
    resumos.append(relatorio_binario(binaria, "Positivas x negativas, todas as fontes"))
    if binaria["fonte"].nunique() > 1:
        for fonte, parte in binaria.groupby("fonte"):
            if parte["rotulo"].nunique() == 2:
                resumos.append(relatorio_binario(parte, f"Somente {fonte}"))
    nao_neutras = binaria["classe_finbert"] != "neutro"
    if 0 < nao_neutras.sum() < len(binaria):
        resumos.append(relatorio_binario(binaria[nao_neutras], "Só onde o FinBERT não respondeu neutro"))
    if (df["rotulo"] == 2).any():
        resumos.append(relatorio_tres_classes(df[df["fonte"] == "notícias com 2 anotadores"]))

    print("\nResumo:")
    print(pd.DataFrame(resumos).round(3).to_string(index=False))

    erros = binaria[binaria["previsto"] != binaria["rotulo"]].copy()
    erros["forca"] = erros["score"].abs()
    print("\nErros com maior confiança (material para a análise crítica):")
    for _, e in erros.sort_values("forca", ascending=False).head(8).iterrows():
        print(f"   real {NOMES[e['rotulo']]:8s} | score {e['score']:+.2f} | {e['texto'][:90]}")

    os.makedirs(os.path.dirname(a.saida) or ".", exist_ok=True)
    df.to_csv(a.saida, index=False)
    pd.DataFrame(resumos).to_csv(a.saida.replace(".csv", "_resumo.csv"), index=False)
    print(f"\n[SALVO] {a.saida} (cada notícia com as probabilidades) e o resumo das métricas")


if __name__ == "__main__":
    main()