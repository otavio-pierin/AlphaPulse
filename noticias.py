"""
Coleta de notícias reais de duas fontes gratuitas e sem chave de API:

1. Google News RSS: busca por janela de datas (operadores after:/before:).
   Devolve no máximo 100 itens por consulta; janelas cheias são divididas ao meio.
2. API REST do WordPress de portais financeiros (Suno Notícias): busca por termo
   e intervalo de datas, paginada de 100 em 100, com histórico de vários anos.

Toda janela concluída é registrada no banco: se a coleta for interrompida, a
próxima execução continua de onde parou. Janelas que terminam hoje nunca são
marcadas como concluídas, para que notícias novas entrem nas próximas execuções.
"""

import hashlib
import html
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

CABECALHOS = {"User-Agent": "Mozilla/5.0 (projeto academico AlphaPulse - UFV)"}


# -----------------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------------
def normalizar(texto):
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


def id_noticia(titulo):
    return hashlib.sha1(normalizar(titulo).encode("utf-8")).hexdigest()[:16]


def limpar_html(texto):
    texto = re.sub(r"<[^>]+>", " ", html.unescape(str(texto or "")))
    return " ".join(texto.split())


def data_efetiva(dt_utc, fuso, hora_corte):
    """Data do pregão em que a notícia já era conhecida no fechamento."""
    local = pd.Timestamp(dt_utc).tz_convert(fuso)
    dia = local.tz_localize(None).normalize()
    if local.hour >= hora_corte:
        dia += pd.Timedelta(days=1)
    return dia.strftime("%Y-%m-%d")


def registro(fonte, veiculo, titulo, resumo, url, dt_utc, fuso, hora_corte):
    titulo = limpar_html(titulo)
    if len(titulo) < 15:
        return None
    return {"id": id_noticia(titulo), "fonte": fonte, "veiculo": veiculo, "titulo": titulo,
            "resumo": limpar_html(resumo)[:600], "url": url,
            "datahora_utc": pd.Timestamp(dt_utc).tz_convert("UTC").isoformat(),
            "data_efetiva": data_efetiva(dt_utc, fuso, hora_corte)}


def get(url, params, tentativas=5, espera=10):
    """
    GET com novas tentativas. Espera progressiva (10, 20, 40, 80 s) em caso de
    bloqueio (429) ou erro do servidor, respeitando o cabeçalho Retry-After.
    Retorna None se todas as tentativas falharem.
    """
    for i in range(tentativas):
        try:
            r = requests.get(url, params=params, headers=CABECALHOS, timeout=30)
        except requests.RequestException as e:
            pausa = espera * 2 ** i
            print(f"   [rede] {type(e).__name__}; nova tentativa em {pausa}s")
            time.sleep(pausa)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            pausa = espera * 2 ** i
            ra = r.headers.get("Retry-After", "")
            if ra.isdigit():
                pausa = max(pausa, int(ra))
            print(f"   [aguardando] HTTP {r.status_code}; nova tentativa em {pausa}s")
            time.sleep(pausa)
            continue
        return r                     # outros códigos (ex.: 400) são tratados por quem chamou
    return None


def janelas(inicio, fim, passo_dias):
    a, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    while a < fim:
        b = min(a + pd.Timedelta(days=passo_dias), fim)
        yield a, b
        a = b


class ColetaInterrompida(RuntimeError):
    pass


def _percorrer(banco, nome, consulta, inicio, fim, passo, buscar_janela):
    """Loop comum às fontes: checkpoint, contagem e parada limpa após falhas seguidas."""
    hoje = pd.Timestamp.today().normalize()
    todas = list(janelas(inicio, fim, passo))
    pendentes = [(a, b) for a, b in todas
                 if not banco.janela_feita(f"{nome}|{consulta}|{a:%Y-%m-%d}|{b:%Y-%m-%d}")]
    print(f"[{nome}] '{consulta}': {len(todas)} janelas, {len(pendentes)} pendentes")

    falhas, novas_total = 0, 0
    for k, (a, b) in enumerate(pendentes, 1):
        regs = buscar_janela(a, b)
        if regs is None:
            falhas += 1
            if falhas >= 3:
                raise ColetaInterrompida(
                    f"[{nome}] 3 janelas seguidas falharam. O que já foi coletado está salvo; "
                    "rode novamente mais tarde para continuar de onde parou.")
            continue
        falhas = 0
        novas = banco.inserir_noticias([r for r in regs if r])
        novas_total += novas
        if b < hoje:                                    # janela fechada: não precisa repetir
            banco.marcar_janela(f"{nome}|{consulta}|{a:%Y-%m-%d}|{b:%Y-%m-%d}", len(regs))
        if k % 10 == 0 or k == len(pendentes):
            print(f"   {k}/{len(pendentes)} | até {b:%Y-%m-%d} | {len(regs)} na janela, "
                  f"{novas_total} novas no total")
    return novas_total


# -----------------------------------------------------------------------------------
# Fonte 1: Google News RSS
# -----------------------------------------------------------------------------------
def coletar_google_news(banco, cfg, inicio, fim):
    g = cfg["google_news"]
    url = "https://news.google.com/rss/search"

    def consultar(a, b):
        params = {"q": f"{g['consulta']} after:{a:%Y-%m-%d} before:{b:%Y-%m-%d}",
                  "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
        r = get(url, params)
        time.sleep(g["pausa"])
        if r is None:
            return None
        if r.status_code != 200:
            return []
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
            if veiculo and titulo.endswith(" - " + veiculo):
                titulo = titulo[: -(len(veiculo) + 3)]
            try:
                dt = pd.Timestamp(parsedate_to_datetime(pub))
            except (TypeError, ValueError):
                continue
            saida.append(registro("google_news", veiculo, titulo, "", item.findtext("link") or "",
                                  dt, cfg["fuso"], cfg["hora_corte"]))
        return saida

    def adaptativo(a, b):
        regs = consultar(a, b)
        if regs is None:
            return None
        dias = (b - a).days
        if len(regs) >= g["limite"] and dias > 1:       # janela cheia: divide ao meio
            meio = a + pd.Timedelta(days=dias // 2)
            r1, r2 = adaptativo(a, meio), adaptativo(meio, b)
            return None if r1 is None or r2 is None else r1 + r2
        return regs

    return _percorrer(banco, "google_news", g["consulta"], inicio, fim, g["passo_dias"], adaptativo)


# -----------------------------------------------------------------------------------
# Fonte 2: API REST do WordPress (Suno Notícias e outros portais)
# -----------------------------------------------------------------------------------
def coletar_wordpress(banco, cfg, inicio, fim):
    w = cfg["wordpress"]
    novas = 0
    for site in w["sites"]:
        for termo in site["termos"]:

            def buscar(a, b, site=site, termo=termo):
                saida, pagina = [], 1
                while True:
                    params = {"search": termo, "after": f"{a:%Y-%m-%d}T00:00:00",
                              "before": f"{b:%Y-%m-%d}T00:00:00", "per_page": 100,
                              "page": pagina, "orderby": "date", "order": "asc",
                              "_fields": "id,date_gmt,title,excerpt,link"}
                    r = get(site["api"], params)
                    time.sleep(w["pausa"])
                    if r is None:
                        return None
                    if r.status_code == 400:            # página além da última
                        break
                    if r.status_code != 200:
                        print(f"   [aviso] {site['nome']}: HTTP {r.status_code}")
                        return None
                    try:
                        posts = r.json()
                    except ValueError:
                        return None
                    for p in posts:
                        dt = pd.to_datetime(p.get("date_gmt"), utc=True, errors="coerce")
                        if pd.isna(dt):
                            continue
                        saida.append(registro("wordpress", site["nome"],
                                              (p.get("title") or {}).get("rendered", ""),
                                              (p.get("excerpt") or {}).get("rendered", ""),
                                              p.get("link", ""), dt, cfg["fuso"], cfg["hora_corte"]))
                    total_paginas = int(r.headers.get("X-WP-TotalPages", "1") or 1)
                    if pagina >= total_paginas or not posts:
                        break
                    pagina += 1
                return saida

            novas += _percorrer(banco, f"wordpress:{site['nome']}", termo, inicio, fim,
                                w["passo_dias"], buscar)
    return novas


def coletar_noticias(banco, cfg, inicio, fim):
    """Executa as fontes ativas. Uma fonte com problema não impede a outra."""
    for nome, funcao in (("google_news", coletar_google_news), ("wordpress", coletar_wordpress)):
        if not cfg[nome]["ativo"]:
            continue
        try:
            funcao(banco, cfg, inicio, fim)
        except ColetaInterrompida as e:
            print(f"[AVISO] {e}")
    print(f"[notícias] Total no banco: {banco.total()} manchetes únicas")