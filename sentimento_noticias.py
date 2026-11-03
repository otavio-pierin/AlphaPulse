"""
=====================================================================================
 sentimento_noticias.py - Coleta de notícias REAIS + score de sentimento
=====================================================================================
 Fontes gratuitas e sem chave de API:
   - Google News RSS  (busca por janela de datas com os operadores after:/before:)
   - GDELT DOC 2.0 API (monitor global de notícias, histórico desde 2017)

 Modelos de sentimento gratuitos:
   - "finbert-pt" -> lucas-leme/FinBERT-PT-BR  (notícias em português, padrão)
   - "finbert-en" -> ProsusAI/finbert          (notícias em inglês)
   - "ollama:<modelo>" -> LLM local via Ollama (ex.: "ollama:qwen2.5:7b"), gratuito,
     roda na sua máquina. A API do GPT (OpenAI) é paga, por isso não é usada.

 Score diário = P(positivo) - P(negativo), no intervalo [-1, 1].

 Instalação:
   pip install requests pandas numpy transformers torch

 Uso isolado (gera dados_noticias/noticias_com_sentimento.csv e sentimento.csv):
   python sentimento_noticias.py --inicio 2018-01-01 --fim 2026-10-19
=====================================================================================
"""

import argparse
import hashlib
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

FUSO_B3 = "America/Sao_Paulo"
CABECALHOS = {"User-Agent": "Mozilla/5.0 (projeto academico AlphaPulse - UFV)"}

MODELOS_HF = {
    "finbert-pt": "lucas-leme/FinBERT-PT-BR",
    "finbert-en": "ProsusAI/finbert",
}

COLUNAS_BRUTAS = ["ID", "DataHora_UTC", "Fonte", "Veiculo", "Titulo", "URL"]


# -----------------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------------
def _normalizar(texto: str) -> str:
    """Minúsculas, sem acento e sem pontuação: usado para deduplicar manchetes."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


def _salvar_csv_atomico(df: pd.DataFrame, caminho: str, **kwargs) -> None:
    """Grava num arquivo temporário e troca de uma vez: nunca deixa um CSV pela metade."""
    tmp = f"{caminho}.{os.getpid()}.tmp"
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, caminho)


def ler_csv_tolerante(caminho: str, **kwargs) -> pd.DataFrame:
    """Lê o CSV; se houver linhas corrompidas, descarta só essas linhas e avisa."""
    try:
        return pd.read_csv(caminho, **kwargs)
    except pd.errors.ParserError:
        df = pd.read_csv(caminho, engine="python", on_bad_lines="skip", **kwargs)
        print(f"   [aviso] {caminho} tinha linhas corrompidas; elas foram ignoradas "
              f"({len(df)} linhas válidas). O arquivo será regravado.")
        return df


def _id_noticia(titulo: str) -> str:
    return hashlib.sha1(_normalizar(titulo).encode("utf-8")).hexdigest()[:16]


def _get(url: str, params: dict, tentativas: int = 4, espera: float = 5.0):
    """GET com novas tentativas e espera crescente (respeita limites de requisição)."""
    for i in range(tentativas):
        try:
            r = requests.get(url, params=params, headers=CABECALHOS, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                motivo = "limite de requisições" if r.status_code == 429 else f"HTTP {r.status_code}"
                pausa = espera * (2 ** i)                       # 1x, 2x, 4x, 8x
                retry_after = r.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    pausa = max(pausa, int(retry_after))
                print(f"   [aguardando] {motivo}: nova tentativa em {pausa:.0f}s")
                time.sleep(pausa)
                continue
            print(f"   [aviso] HTTP {r.status_code} em {url}")
            return None
        except requests.RequestException as e:
            pausa = espera * (2 ** i)
            print(f"   [aviso] {type(e).__name__}: nova tentativa em {pausa:.0f}s")
            time.sleep(pausa)
    return None


def _janelas(inicio, fim, passo_dias):
    atual, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    while atual < fim:
        prox = min(atual + pd.Timedelta(days=passo_dias), fim)
        yield atual, prox
        atual = prox


def _coletar_adaptativo(funcao_janela, a, b, limite, pausa, contador):
    """
    Busca a janela [a, b). Se o resultado atingir o limite da fonte (ex.: 250 no
    GDELT, 100 no Google News), parte da janela foi cortada; então ela é dividida
    ao meio e cada metade é buscada de novo, até chegar a 1 dia. Isso evita que
    alguns dias da semana fiquem sistematicamente sem notícias.
    """
    registros = funcao_janela(a, b)
    contador["req"] += 1
    time.sleep(pausa)
    if registros is None:
        return None
    dias = (b - a).days
    if limite and len(registros) >= limite and dias > 1:
        meio = a + pd.Timedelta(days=dias // 2)
        r1 = _coletar_adaptativo(funcao_janela, a, meio, limite, pausa, contador)
        if r1 is None:
            return None
        r2 = _coletar_adaptativo(funcao_janela, meio, b, limite, pausa, contador)
        if r2 is None:
            return None
        return r1 + r2
    if limite and len(registros) >= limite:
        contador["dias_no_limite"] += 1
    return registros


def _coletar_com_checkpoint(nome_fonte, funcao_janela, inicio, fim, passo_dias, pausa, pasta,
                            limite=None):
    """
    Percorre o período em janelas (divididas automaticamente quando atingem o
    limite da fonte) e grava cada janela assim que termina. Se a coleta for
    interrompida, basta rodar de novo: as janelas já baixadas são puladas
    (checkpoint em <pasta>/<fonte>_janelas.txt).
    """
    os.makedirs(pasta, exist_ok=True)
    arq_dados = os.path.join(pasta, f"{nome_fonte}_bruto.csv")
    arq_ok = os.path.join(pasta, f"{nome_fonte}_janelas.txt")
    feitas = set(open(arq_ok, encoding="utf-8").read().split()) if os.path.exists(arq_ok) else set()

    janelas = list(_janelas(inicio, fim, passo_dias))
    pendentes = [(a, b) for a, b in janelas if f"{a:%Y%m%d}-{b:%Y%m%d}" not in feitas]
    print(f"[{nome_fonte}] {len(janelas)} janelas no período, {len(pendentes)} pendentes")

    contador = {"req": 0, "dias_no_limite": 0}
    falhas_seguidas = 0
    for k, (a, b) in enumerate(pendentes, 1):
        registros = _coletar_adaptativo(funcao_janela, a, b, limite, pausa, contador)
        if registros is None:          # falha de rede: não marca a janela como feita
            falhas_seguidas += 1
            if falhas_seguidas >= 3:
                restantes = len(pendentes) - k + 1
                raise RuntimeError(
                    f"[{nome_fonte}] O servidor está bloqueando as requisições (3 janelas "
                    f"seguidas falharam). {restantes} janelas continuam pendentes. "
                    f"Aguarde 30-60 minutos e rode de novo: a coleta retoma de onde parou.")
            print(f"   [aviso] janela {a:%Y-%m-%d} falhou; pausa de 2 min antes de seguir")
            time.sleep(120)
            continue
        falhas_seguidas = 0
        if registros:
            df = pd.DataFrame(registros, columns=COLUNAS_BRUTAS)
            df.to_csv(arq_dados, mode="a", index=False, header=not os.path.exists(arq_dados))
        with open(arq_ok, "a", encoding="utf-8") as f:
            f.write(f"{a:%Y%m%d}-{b:%Y%m%d}\n")
        if k % 5 == 0 or k == len(pendentes):
            print(f"   {k}/{len(pendentes)} janelas | última: {a:%Y-%m-%d} "
                  f"({len(registros)} notícias) | requisições: {contador['req']}")

    if contador["dias_no_limite"]:
        print(f"   [info] {contador['dias_no_limite']} dia(s) ainda no limite mesmo com janela "
              f"de 1 dia (consulta ampla demais para esses dias)")

    if not os.path.exists(arq_dados):
        return pd.DataFrame(columns=COLUNAS_BRUTAS)
    return ler_csv_tolerante(arq_dados)


# -----------------------------------------------------------------------------------
# Fonte 1: Google News RSS
# -----------------------------------------------------------------------------------
def coletar_google_news(consulta, inicio, fim, pasta, passo_dias=7, pausa=1.5):
    """Até ~100 manchetes por consulta; janelas semanais cobrem bem o histórico."""
    url = "https://news.google.com/rss/search"

    def por_janela(a, b):
        params = {"q": f"{consulta} after:{a:%Y-%m-%d} before:{b:%Y-%m-%d}",
                  "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
        r = _get(url, params)
        if r is None:
            return None
        try:
            raiz = ET.fromstring(r.content)
        except ET.ParseError:
            return []
        saida = []
        for item in raiz.iter("item"):
            titulo = (item.findtext("title") or "").strip()
            veiculo = (item.findtext("source") or "").strip()
            pub = item.findtext("pubDate")
            if not titulo or not pub:
                continue
            if veiculo and titulo.endswith(" - " + veiculo):     # remove " - Veículo"
                titulo = titulo[: -(len(veiculo) + 3)].strip()
            try:
                dt = pd.Timestamp(parsedate_to_datetime(pub)).tz_convert("UTC")
            except (TypeError, ValueError):
                continue
            saida.append({"ID": _id_noticia(titulo), "DataHora_UTC": dt.isoformat(),
                          "Fonte": "google_news", "Veiculo": veiculo,
                          "Titulo": titulo, "URL": item.findtext("link") or ""})
        return saida

    return _coletar_com_checkpoint("google_news", por_janela, inicio, fim, passo_dias, pausa, pasta,
                                   limite=100)


# -----------------------------------------------------------------------------------
# Fonte 2: GDELT DOC 2.0
# -----------------------------------------------------------------------------------
def coletar_gdelt(consulta, inicio, fim, pasta, passo_dias=7, pausa=8.0):
    """
    Até 250 artigos por consulta. O GDELT pede no máximo 1 requisição a cada
    5 segundos; usamos 8s de folga, pois as divisões de janela aumentam o número de
    requisições. Histórico a partir de 2017.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    inicio = max(pd.Timestamp(inicio), pd.Timestamp("2017-01-01"))

    def por_janela(a, b):
        params = {"query": consulta, "mode": "artlist", "format": "json",
                  "maxrecords": 250, "sort": "datedesc",
                  "startdatetime": f"{a:%Y%m%d}000000", "enddatetime": f"{b:%Y%m%d}000000"}
        r = _get(url, params, espera=30)          # 30s, 60s, 120s, 240s
        if r is None:
            return None
        try:
            artigos = r.json().get("articles", [])
        except ValueError:            # GDELT devolve texto quando a consulta é inválida
            print(f"   [aviso] GDELT: {r.text[:120]}")
            return []
        saida = []
        for art in artigos:
            titulo = (art.get("title") or "").strip()
            if not titulo:
                continue
            dt = pd.to_datetime(art.get("seendate"), format="%Y%m%dT%H%M%SZ", utc=True,
                                errors="coerce")
            if pd.isna(dt):
                continue
            saida.append({"ID": _id_noticia(titulo), "DataHora_UTC": dt.isoformat(),
                          "Fonte": "gdelt", "Veiculo": art.get("domain", ""),
                          "Titulo": titulo, "URL": art.get("url", "")})
        return saida

    return _coletar_com_checkpoint("gdelt", por_janela, inicio, fim, passo_dias, pausa, pasta,
                                   limite=250)


# -----------------------------------------------------------------------------------
# Limpeza e data efetiva (anti data leakage)
# -----------------------------------------------------------------------------------
def preparar_noticias(brutas: pd.DataFrame, hora_corte: int = 17, min_caracteres: int = 15):
    """
    - Remove manchetes curtas e duplicadas (mesma manchete em vários veículos/dias:
      fica a ocorrência mais antiga, que é quando a informação chegou ao mercado).
    - Converte o horário para Brasília e calcula a 'Data' efetiva: notícias
      publicadas a partir de 'hora_corte' (após o fechamento da B3) passam para o
      dia seguinte. Assim o modelo só usa no dia t o que era conhecido até o
      fechamento de t. Fins de semana/feriados são tratados depois, no merge.
    """
    df = brutas.dropna(subset=["Titulo", "DataHora_UTC"]).copy()
    # Quebras de linha e espaços repetidos viram um espaço só
    df["Titulo"] = df["Titulo"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["Titulo"].str.len() >= min_caracteres]

    df["DataHora_UTC"] = pd.to_datetime(df["DataHora_UTC"], utc=True, errors="coerce")
    df = df.dropna(subset=["DataHora_UTC"])
    df["ID"] = df["Titulo"].map(_id_noticia)
    df = df.sort_values("DataHora_UTC").drop_duplicates("ID", keep="first")

    local = df["DataHora_UTC"].dt.tz_convert(FUSO_B3)
    df["DataHora_Brasilia"] = local.dt.strftime("%Y-%m-%d %H:%M")
    df["Data"] = local.dt.tz_localize(None).dt.normalize()
    apos_fechamento = (local.dt.hour >= hora_corte).astype(int)
    df["Data"] = df["Data"] + pd.to_timedelta(apos_fechamento, unit="D")
    df["Apos_Fechamento"] = apos_fechamento
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------------
# Modelos de sentimento
# -----------------------------------------------------------------------------------
class AnalisadorFinBERT:
    """FinBERT (Hugging Face). Baixa o modelo na 1ª execução (~400 MB) e usa GPU se houver."""

    def __init__(self, modelo: str = "finbert-pt", batch: int = 32, max_tokens: int = 96):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.nome = MODELOS_HF.get(modelo, modelo)
        self.batch, self.max_tokens = batch, max_tokens
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sentimento] Carregando {self.nome} em {self.dispositivo}...")
        self.tok = AutoTokenizer.from_pretrained(self.nome)
        self.modelo = self._carregar_modelo(AutoModelForSequenceClassification)
        self.modelo.to(self.dispositivo).eval()

        # Descobre a posição de cada rótulo (os modelos usam ordens diferentes)
        rotulos = {i: str(r).lower() for i, r in self.modelo.config.id2label.items()}
        achar = lambda chave: next((i for i, r in rotulos.items() if chave in r), None)
        self.i_pos, self.i_neg, self.i_neu = achar("pos"), achar("neg"), achar("neu")
        if self.i_pos is None or self.i_neg is None:
            raise ValueError(f"Rótulos inesperados no modelo {self.nome}: {rotulos}")

    def _carregar_modelo(self, classe):
        """
        Tenta primeiro o formato safetensors (seguro, funciona com qualquer PyTorch).
        Se o modelo só existir no formato antigo (.bin), o transformers exige
        PyTorch >= 2.6 por causa da vulnerabilidade CVE-2025-32434.
        """
        try:
            return classe.from_pretrained(self.nome, use_safetensors=True)
        except (OSError, ValueError):
            pass
        try:
            return classe.from_pretrained(self.nome)
        except ValueError as e:
            if "2.6" in str(e):
                raise RuntimeError(
                    f"O modelo {self.nome} só está disponível no formato .bin, que exige "
                    f"PyTorch >= 2.6 (versão instalada: {self.torch.__version__}). Atualize com:\n"
                    '   pip install --upgrade "torch>=2.6" --index-url '
                    "https://download.pytorch.org/whl/cu124") from e
            raise

    def pontuar(self, textos):
        probs = []
        with self.torch.no_grad():
            for i in range(0, len(textos), self.batch):
                lote = [str(t) for t in textos[i: i + self.batch]]
                enc = self.tok(lote, padding=True, truncation=True,
                               max_length=self.max_tokens, return_tensors="pt").to(self.dispositivo)
                logits = self.modelo(**enc).logits
                probs.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
                if (i // self.batch) % 50 == 0:
                    print(f"   pontuadas {min(i + self.batch, len(textos))}/{len(textos)}")
        p = np.vstack(probs) if probs else np.zeros((0, len(self.modelo.config.id2label)))
        saida = pd.DataFrame({
            "Prob_Positivo": p[:, self.i_pos],
            "Prob_Negativo": p[:, self.i_neg],
            "Prob_Neutro": p[:, self.i_neu] if self.i_neu is not None else np.nan,
        })
        saida["Sentimento_Score"] = saida["Prob_Positivo"] - saida["Prob_Negativo"]
        saida["Modelo_Sentimento"] = self.nome
        return saida


class AnalisadorLLMLocal:
    """
    LLM local e gratuito via Ollama (https://ollama.com). Ex.: 'ollama pull qwen2.5:7b'.
    Bem mais lento que o FinBERT; indicado para bases menores ou para comparar modelos.
    """

    PROMPT = ("Você é um analista do mercado financeiro brasileiro. Avalie o impacto da "
              "manchete no mercado de ações. Responda APENAS um JSON no formato "
              '{"score": x}, com x entre -1 (muito negativo) e 1 (muito positivo); '
              "0 é neutro.")

    def __init__(self, modelo="qwen2.5:7b", url="http://localhost:11434/api/chat"):
        self.modelo, self.url = modelo, url

    def pontuar(self, textos):
        import json
        scores = []
        for i, t in enumerate(textos, 1):
            corpo = {"model": self.modelo, "stream": False, "format": "json",
                     "options": {"temperature": 0},
                     "messages": [{"role": "system", "content": self.PROMPT},
                                  {"role": "user", "content": str(t)}]}
            try:
                r = requests.post(self.url, json=corpo, timeout=120)
                score = float(json.loads(r.json()["message"]["content"])["score"])
            except Exception:
                score = np.nan
            scores.append(float(np.clip(score, -1, 1)) if not np.isnan(score) else np.nan)
            if i % 100 == 0:
                print(f"   pontuadas {i}/{len(textos)}")
        return pd.DataFrame({"Prob_Positivo": np.nan, "Prob_Negativo": np.nan,
                             "Prob_Neutro": np.nan, "Sentimento_Score": scores,
                             "Modelo_Sentimento": f"ollama:{self.modelo}"})


def criar_analisador(modelo: str):
    if modelo.startswith("ollama:"):
        return AnalisadorLLMLocal(modelo.split(":", 1)[1])
    return AnalisadorFinBERT(modelo)


# -----------------------------------------------------------------------------------
# Pipeline completo
# -----------------------------------------------------------------------------------
def construir_base_sentimento(inicio, fim=None,
                              fontes=("google_news", "gdelt"),
                              consulta_google_news='Ibovespa OR "Bolsa brasileira"',
                              consulta_gdelt='(ibovespa OR "bolsa de valores" OR "bolsa brasileira") sourcelang:portuguese',
                              modelo="finbert-pt", pasta="dados_noticias",
                              hora_corte=17, saida_sentimento="sentimento.csv"):
    """
    1. Coleta notícias reais (com checkpoint)   2. Limpa, deduplica e define a data efetiva
    3. Pontua com FinBERT/LLM (reaproveita scores já calculados)   4. Salva os CSVs
    """
    fim = fim or pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    os.makedirs(pasta, exist_ok=True)
    brutas = []
    if "google_news" in fontes:
        brutas.append(coletar_google_news(consulta_google_news, inicio, fim, pasta))
    if "gdelt" in fontes:
        brutas.append(coletar_gdelt(consulta_gdelt, inicio, fim, pasta))

    brutas = pd.concat(brutas, ignore_index=True) if brutas else pd.DataFrame(columns=COLUNAS_BRUTAS)
    if brutas.empty:
        raise RuntimeError("Nenhuma notícia coletada. Verifique a conexão ou as consultas.")

    noticias = preparar_noticias(brutas, hora_corte=hora_corte)
    print(f"[sentimento] {len(noticias)} manchetes únicas após limpeza")

    # Cache: só pontua manchetes novas (útil ao ampliar o período ou rodar de novo)
    arq_final = os.path.join(pasta, "noticias_com_sentimento.csv")
    colunas_score = ["Prob_Positivo", "Prob_Negativo", "Prob_Neutro",
                     "Sentimento_Score", "Modelo_Sentimento"]
    if os.path.exists(arq_final):
        anteriores = ler_csv_tolerante(arq_final)
        anteriores = anteriores[[c for c in ["ID"] + colunas_score if c in anteriores.columns]]
        nome_modelo = MODELOS_HF.get(modelo, modelo)
        anteriores = anteriores[anteriores["Modelo_Sentimento"].astype(str) == nome_modelo]
        noticias = noticias.merge(anteriores.drop_duplicates("ID"), on="ID", how="left")
    else:
        for c in colunas_score:
            noticias[c] = np.nan
    noticias["Modelo_Sentimento"] = noticias["Modelo_Sentimento"].astype(object)

    faltam = noticias["Sentimento_Score"].isna()
    if faltam.any():
        print(f"[sentimento] Pontuando {faltam.sum()} manchetes com '{modelo}'...")
        scores = criar_analisador(modelo).pontuar(noticias.loc[faltam, "Titulo"].tolist())
        for c in colunas_score:               # coluna a coluna preserva os tipos
            noticias.loc[faltam, c] = scores[c].to_numpy()

    noticias = noticias.dropna(subset=["Sentimento_Score"])
    colunas_saida = ["ID", "Data", "DataHora_Brasilia", "Apos_Fechamento", "Fonte", "Veiculo",
                     "Titulo", "URL"] + colunas_score
    noticias = noticias.sort_values(["Data", "DataHora_Brasilia"])[colunas_saida]
    _salvar_csv_atomico(noticias, arq_final, index=False, float_format="%.6f")

    sentimento = noticias[["Data", "Sentimento_Score", "Titulo", "Fonte", "Veiculo"]]
    _salvar_csv_atomico(sentimento, saida_sentimento, index=False, float_format="%.6f")

    print(f"[SALVO] {arq_final} ({len(noticias)} notícias com sentimento)")
    print(f"[SALVO] {saida_sentimento} (entrada do modelo de previsão)")
    print(f"[sentimento] Score médio {noticias['Sentimento_Score'].mean():+.3f} | "
          f"dias com notícia: {noticias['Data'].nunique()}")
    return sentimento


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Coleta notícias reais e calcula sentimento.")
    ap.add_argument("--inicio", default="2018-01-01")
    ap.add_argument("--fim", default=None)
    ap.add_argument("--modelo", default="finbert-pt",
                    help="finbert-pt | finbert-en | ollama:<modelo>")
    ap.add_argument("--fontes", default="google_news,gdelt")
    ap.add_argument("--pasta", default="dados_noticias")
    ap.add_argument("--saida", default="sentimento.csv")
    a = ap.parse_args()
    construir_base_sentimento(a.inicio, a.fim, tuple(a.fontes.split(",")),
                              modelo=a.modelo, pasta=a.pasta, saida_sentimento=a.saida)