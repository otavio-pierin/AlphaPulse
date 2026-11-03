"""
AlphaPulse - aplicação web
Mostra, para a ação escolhida, os dados do último fechamento, a previsão de
direção para o próximo pregão e as manchetes que apontam na mesma direção.

Executar:
    pip install streamlit
    streamlit run app.py
"""

import os
import threading
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import main as M
import sentimento_noticias as SN
from ativos import ATIVOS, caminho_modelo, caminho_noticias, config_ativo

st.set_page_config(page_title="AlphaPulse", page_icon="📈", layout="wide")

FUSO = "America/Sao_Paulo"
PREGOES_NOTICIAS = 3      # mesmo horizonte da feature Sent_MM3 usada pelo modelo
LIMIAR_SENTIMENTO = 0.15  # |score| abaixo disso é tratado como neutro
COR_ALTA, COR_BAIXA = "#1B7F5B", "#B3392F"

st.markdown(f"""
<style>
  .veredito {{ border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid; }}
  .veredito .rotulo {{ font-size: .95rem; opacity: .8; margin: 0; }}
  .veredito .direcao {{ font-size: 3.2rem; font-weight: 700; line-height: 1.05; margin: .2rem 0 .6rem; }}
  .veredito .barra {{ height: 10px; border-radius: 6px; background: {COR_BAIXA}33; overflow: hidden; }}
  .veredito .barra > div {{ height: 100%; background: {COR_ALTA}; }}
  .veredito .escala {{ display: flex; justify-content: space-between; font-size: .8rem; opacity: .75; margin-top: .3rem; }}
  .veredito .detalhe {{ font-size: .95rem; margin: .8rem 0 0; }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------------
# Formatação pt-BR
# -----------------------------------------------------------------------------------
def br(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def br_compacto(x):
    for lim, suf in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
        if abs(x) >= lim:
            return br(x / lim, 1) + suf
    return br(x, 0)


# -----------------------------------------------------------------------------------
# Recursos em cache (carregados uma vez por sessão do servidor)
# -----------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def analisador_finbert(nome):
    return SN.AnalisadorFinBERT(nome)


def _criar_analisador(modelo):
    if modelo.startswith("ollama:"):
        return SN.AnalisadorLLMLocal(modelo.split(":", 1)[1])
    return analisador_finbert(modelo)


SN.criar_analisador = _criar_analisador   # evita recarregar o FinBERT a cada consulta


@st.cache_resource(show_spinner=False)
def carregar_modelo(caminho, mtime):
    """mtime entra na chave do cache: um modelo retreinado é recarregado."""
    return M.carregar_modelo_pth(caminho)


@st.cache_resource(show_spinner=False)
def trava_global():
    # Compartilhada por todas as execuções do servidor: evita que duas análises
    # gravem os mesmos CSVs ao mesmo tempo (o Streamlit pode iniciar uma nova
    # execução enquanto a anterior ainda está terminando em segundo plano).
    return threading.Lock()


# -----------------------------------------------------------------------------------
# Previsão
# -----------------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def analisar(ticker, versao):
    """'versao' muda quando o usuário pede atualização, invalidando o cache."""
    with trava_global():
        return _analisar(ticker)


def _analisar(ticker):
    cfg = config_ativo(ticker)
    pth = caminho_modelo(cfg)
    modelo, scaler, ck = carregar_modelo(pth, os.path.getmtime(pth))

    precos = M.baixar_precos(ticker, cfg["inicio"], None)

    # Durante o pregão o yfinance devolve o candle de hoje ainda incompleto
    agora = pd.Timestamp.now(tz=FUSO)
    em_andamento = precos.index[-1].date() == agora.date() and agora.hour < 18
    if em_andamento:
        precos = precos.iloc[:-1]

    aviso = None
    try:
        SN.construir_base_sentimento(
            inicio=precos.index.min(), fim=agora.tz_localize(None).normalize() + pd.Timedelta(days=1),
            fontes=cfg["fontes_noticias"], consulta_google_news=cfg["consulta_google_news"],
            consulta_gdelt=cfg["consulta_gdelt"], modelo=cfg["modelo_sentimento"],
            pasta=cfg["pasta_noticias"], hora_corte=cfg["hora_corte_pregao"],
            saida_sentimento=cfg["csv_sentimento"])
    except Exception as e:
        aviso = f"Não foi possível buscar notícias novas ({e}). A previsão usa as notícias já salvas."

    sentimento = M.carregar_sentimento(precos, cfg)
    dados = M.criar_features(M.alinhar_dados(precos, sentimento))

    J, fp, fs = ck["janela"], ck["features_preco"], ck["features_sent"]
    X = scaler.transform(dados[fp + fs].iloc[-J:]).astype("float32")[None, ...]
    p_alta = float(modelo.predict([X[:, :, :len(fp)], X[:, :, len(fp):]], verbose=0)[0, 0])

    # Notícias dos últimos pregões (as que entraram no modelo) e posteriores ao fechamento
    noticias = SN.ler_csv_tolerante(caminho_noticias(cfg), parse_dates=["Data"])
    pregoes = dados.index
    ultimo = pregoes[-1]
    inicio_janela = pregoes[-PREGOES_NOTICIAS - 1] if len(pregoes) > PREGOES_NOTICIAS else pregoes[0]
    usadas = noticias[(noticias["Data"] > inicio_janela) & (noticias["Data"] <= ultimo)]
    posteriores = noticias[noticias["Data"] > ultimo]

    return {
        "precos": precos.tail(120), "p_alta": p_alta, "ultimo": ultimo,
        "usadas": usadas, "posteriores": posteriores, "aviso": aviso,
        "em_andamento": em_andamento, "atualizado": agora,
    }


# -----------------------------------------------------------------------------------
# Componentes de interface
# -----------------------------------------------------------------------------------
def grafico_fechamento(serie):
    """
    Gráfico renderizado como imagem no servidor (matplotlib), sem depender dos
    componentes JavaScript de gráfico do Streamlit. Fundo transparente e cinza
    neutro nos eixos: fica legível tanto no tema claro quanto no escuro.
    """
    neutro = "#8A94A0"
    cor = COR_ALTA if serie.iloc[-1] >= serie.iloc[0] else COR_BAIXA
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.plot(serie.index, serie.values, color=cor, linewidth=1.6)
    ax.fill_between(serie.index, serie.values, serie.min(), color=cor, alpha=0.10)
    ax.scatter([serie.index[-1]], [serie.iloc[-1]], color=cor, s=18, zorder=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: br(v, 0)))
    ax.tick_params(colors=neutro, labelsize=7)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(neutro)
    ax.grid(axis="y", color=neutro, alpha=0.2, linewidth=0.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)


def bloco_veredito(p_alta, ultimo):
    alta = p_alta >= 0.5
    cor = COR_ALTA if alta else COR_BAIXA
    forca = abs(p_alta - 0.5)
    intensidade = "sinal fraco" if forca < 0.05 else "sinal moderado" if forca < 0.10 else "sinal forte"
    proximo = (ultimo + pd.offsets.BDay(1)).strftime("%d/%m/%Y")
    st.markdown(f"""
    <div class="veredito" style="border-color:{cor}55; background:{cor}12;">
      <p class="rotulo">Previsão para o próximo pregão ({proximo})</p>
      <p class="direcao" style="color:{cor};">{"▲ Alta" if alta else "▼ Baixa"}</p>
      <div class="barra"><div style="width:{p_alta * 100:.1f}%"></div></div>
      <div class="escala"><span>Baixa</span><span>Alta</span></div>
      <p class="detalhe">Probabilidade de alta: <b>{br(p_alta * 100, 1)}%</b>, {intensidade}.</p>
    </div>""", unsafe_allow_html=True)


def card_noticia(linha):
    score = linha["Sentimento_Score"]
    cor = "green" if score >= LIMIAR_SENTIMENTO else "red" if score <= -LIMIAR_SENTIMENTO else "gray"
    tom = "Positiva" if cor == "green" else "Negativa" if cor == "red" else "Neutra"
    with st.container(border=True):
        titulo = str(linha["Titulo"])
        url = str(linha.get("URL", "") or "")
        st.markdown(f"**[{titulo}]({url})**" if url.startswith("http") else f"**{titulo}**")
        quando = str(linha.get("DataHora_Brasilia", "") or "")
        veiculo = str(linha.get("Veiculo", "") or "")
        st.caption(f"{veiculo}, {quando}" if veiculo else quando)
        st.markdown(f":{cor}[{tom} ({'+' if score > 0 else ''}{br(score, 2)})]")


def grade_noticias(df, colunas=3):
    for i in range(0, len(df), colunas):
        cols = st.columns(colunas)
        for c, (_, linha) in zip(cols, df.iloc[i:i + colunas].iterrows()):
            with c:
                card_noticia(linha)


# -----------------------------------------------------------------------------------
# Página
# -----------------------------------------------------------------------------------
if "versao" not in st.session_state:
    st.session_state.versao = 0

with st.sidebar:
    st.title("AlphaPulse")
    st.caption("Previsão da direção do próximo pregão com preços e sentimento de notícias.")
    ticker = st.selectbox("Ação", list(ATIVOS), format_func=lambda t: f"{ATIVOS[t]['nome']} ({t})")
    if st.button("Atualizar dados", use_container_width=True):
        st.session_state.versao += 1
    st.divider()
    st.caption("Modelos disponíveis")
    for t, info in ATIVOS.items():
        ok = os.path.exists(caminho_modelo(config_ativo(t)))
        st.caption(f"{'●' if ok else '○'} {info['nome']}")
    st.divider()
    st.caption("Projeto acadêmico. As previsões não são recomendação de investimento.")

cfg = config_ativo(ticker)
nome = ATIVOS[ticker]["nome"]
st.header(f"{nome} ({ticker})")

if not os.path.exists(caminho_modelo(cfg)):
    st.info(f"Ainda não existe um modelo treinado para {nome}. O treino coleta as notícias da "
            "empresa, calcula o sentimento e treina a rede: leva de 20 a 30 minutos.")
    st.code(f"python preparar_ativos.py {ticker}", language="bash")
    if st.button(f"Treinar modelo de {nome} agora"):
        with st.status("Treinando... acompanhe o progresso no terminal.", expanded=True):
            with trava_global():
                M.main(cfg)
        st.cache_data.clear()
        st.rerun()
    st.stop()

try:
    with st.spinner("Buscando preços e notícias recentes..."):
        r = analisar(ticker, st.session_state.versao)
except Exception as e:
    st.error(f"Não foi possível gerar a previsão: {e}")
    st.stop()

if r["aviso"]:
    st.warning(r["aviso"])
if r["em_andamento"]:
    st.caption("O pregão de hoje ainda está aberto; a análise usa o último fechamento completo.")

# --- Último fechamento + previsão ---
precos = r["precos"]
u, ant = precos.iloc[-1], precos.iloc[-2]
var = (u["Close"] / ant["Close"] - 1) * 100

esq, dir_ = st.columns([3, 2], gap="large")
with esq:
    st.subheader(f"Último fechamento em {r['ultimo']:%d/%m/%Y}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fechamento", br(u["Close"]), f"{br(var)}%")
    c2.metric("Máxima", br(u["High"]))
    c3.metric("Mínima", br(u["Low"]))
    c4, c5, c6 = st.columns(3)
    c4.metric("Abertura", br(u["Open"]))
    c5.metric("Volume", br_compacto(u["Volume"]))
    c6.metric("Fechamento anterior", br(ant["Close"]))
    st.caption("Fechamento nos últimos 120 pregões")
    grafico_fechamento(precos["Close"])
with dir_:
    bloco_veredito(r["p_alta"], r["ultimo"])
    st.caption("A previsão combina os últimos 60 pregões de preço com o sentimento das notícias, "
               f"avaliado pelo FinBERT-PT-BR. Atualizado às {r['atualizado']:%H:%M}.")

# --- Manchetes ---
st.divider()
alta = r["p_alta"] >= 0.5
usadas = r["usadas"].copy()
sinal = 1 if alta else -1
usadas["alinhamento"] = usadas["Sentimento_Score"] * sinal
a_favor = usadas[usadas["alinhamento"] >= LIMIAR_SENTIMENTO].sort_values("alinhamento", ascending=False)
contra = usadas[usadas["alinhamento"] <= -LIMIAR_SENTIMENTO].sort_values("alinhamento")
neutras = len(usadas) - len(a_favor) - len(contra)

st.subheader(f"Manchetes que apontam para {'alta' if alta else 'baixa'}")
if usadas.empty:
    st.info(f"Não houve notícias sobre {nome} nos últimos {PREGOES_NOTICIAS} pregões. "
            "Nesse caso a previsão se apoia apenas no comportamento do preço.")
else:
    st.caption(f"Nos últimos {PREGOES_NOTICIAS} pregões: {len(a_favor)} notícias na mesma direção da previsão, "
               f"{len(contra)} em sentido contrário e {neutras} neutras. As manchetes mostram o contexto do "
               "sentimento; o modelo também considera o histórico de preços.")
    if a_favor.empty:
        st.write("Nenhuma manchete do período aponta na mesma direção; a previsão vem principalmente do preço.")
    else:
        grade_noticias(a_favor.head(9))
    if not contra.empty:
        with st.expander(f"Manchetes em sentido contrário ({len(contra)})"):
            grade_noticias(contra.head(9))

if not r["posteriores"].empty:
    with st.expander(f"Publicadas após o último fechamento ({len(r['posteriores'])}), ainda não usadas pelo modelo"):
        grade_noticias(r["posteriores"].sort_values("Sentimento_Score", key=abs, ascending=False).head(9))