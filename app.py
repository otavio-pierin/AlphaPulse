"""
AlphaPulse - aplicação web (PoC)

Funcionalidade principal (validada): termômetro de sentimento das notícias financeiras
com o FinBERT-PT-BR ajustado, e análise de qualquer manchete digitada.
Sinais experimentais: direção e risco do próximo pregão, exibidos com o desempenho
medido em walk-forward (próximo do acaso, como a literatura indica).

Executar na pasta do projeto:
    pip install streamlit
    streamlit run app.py
"""

import copy
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

import servico_app as S
from config import CONFIG

st.set_page_config(page_title="AlphaPulse", page_icon="📈", layout="wide")

VERDE, VERMELHO, CINZA, AZUL = "#1B7F5B", "#B3392F", "#8A94A0", "#2E6DA4"
st.markdown(f"""
<style>
  .cartao {{ border-radius: 14px; padding: 1.1rem 1.3rem; border: 1px solid; margin-bottom: .6rem; }}
  .cartao .rotulo {{ font-size: .9rem; opacity: .8; margin: 0; }}
  .cartao .valor {{ font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin: .2rem 0 .4rem; }}
  .cartao .detalhe {{ font-size: .88rem; margin: .2rem 0 0; opacity: .9; }}
  .barra {{ height: 9px; border-radius: 6px; background: {VERMELHO}33; overflow: hidden; margin-top: .4rem; }}
  .barra > div {{ height: 100%; background: {VERDE}; }}
  .selo {{ display:inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .78rem;
           font-weight: 600; border: 1px solid; }}
</style>""", unsafe_allow_html=True)


def br(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, casas=1):
    return br(x * 100, casas) + "%"


def cartao(rotulo, valor, detalhe, cor, barra=None):
    b = f'<div class="barra"><div style="width:{barra * 100:.1f}%"></div></div>' if barra is not None else ""
    st.markdown(f"""<div class="cartao" style="border-color:{cor}66; background:{cor}12;">
      <p class="rotulo">{rotulo}</p><p class="valor" style="color:{cor};">{valor}</p>{b}
      <p class="detalhe">{detalhe}</p></div>""", unsafe_allow_html=True)


def eixos_neutros(ax):
    ax.set_facecolor("none")
    ax.tick_params(colors=CINZA, labelsize=7)
    for lado in ("top",):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom", "right"):
        ax.spines[lado].set_color(CINZA)
    ax.grid(axis="y", color=CINZA, alpha=0.2, linewidth=0.6)


# -----------------------------------------------------------------------------------
# Recursos em cache
# -----------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def trava():
    return threading.Lock()          # uma atualização de dados por vez


@st.cache_resource(show_spinner=False)
def carregar_finbert(nome):
    from sentimento import FinBERT
    f = CONFIG["finbert"]
    return FinBERT(nome, f["batch"], f["max_tokens"])


@st.cache_data(show_spinner=False)
def dados(versao):
    return S.ler_base(), S.ler_noticias(), S.metricas_salvas()


@st.cache_data(show_spinner=False)
def sinal(alvo, versao):
    try:
        return S.prever_sinal(alvo), None
    except Exception as e:                       # o app continua funcionando sem os sinais
        return None, str(e)


def atualizar_dados():
    import gerar_base
    with trava():
        gerar_base.main(copy.deepcopy(CONFIG))


# -----------------------------------------------------------------------------------
# Barra lateral
# -----------------------------------------------------------------------------------
if "versao" not in st.session_state:
    st.session_state.versao = 0

info = S.info_modelo_sentimento()
with st.sidebar:
    st.title("AlphaPulse")
    st.caption("Monitor de sentimento das notícias financeiras do Ibovespa, com IA validada.")
    if st.button("Atualizar preços e notícias", use_container_width=True):
        with st.status("Atualizando: preços, notícias novas e sentimento...", expanded=False):
            atualizar_dados()
        st.cache_data.clear()
        st.session_state.versao += 1
        st.rerun()
    st.divider()
    st.caption("Modelo de sentimento")
    st.caption(("FinBERT-PT-BR ajustado (fine-tuning)" if info["ajustado"] else "FinBERT-PT-BR original") +
               f"\n\n`{info['nome']}`")
    st.divider()
    st.caption("Projeto acadêmico (SIN 323 - UFV). Não é recomendação de investimento.")

base, noticias, tabelas = dados(st.session_state.versao)
if base is None or noticias is None:
    st.error("Base de dados não encontrada. Rode primeiro: `python gerar_base.py`")
    st.stop()

fb = S.resumo_finbert(tabelas)
st.header("AlphaPulse · Ibovespa")
if not info["ajustado"]:
    st.warning("O app está usando o FinBERT original. Para usar o ajustado (88% de acerto), defina "
               '`"modelo": "modelos_finbert/finbert-pt-br-ft"` em config.py e rode `python gerar_base.py`.')
elif "modelo" in noticias.columns and not noticias["modelo"].astype(str).eq(info["nome"]).all():
    st.warning("Parte das manchetes foi pontuada com outro modelo. Clique em **Atualizar preços e notícias**.")

aba_sent, aba_manchete, aba_sinais, aba_valid, aba_limites = st.tabs(
    ["📰 Sentimento", "🔎 Analisar manchete", "📈 Sinais experimentais", "✅ Validação", "⚠️ Limitações"])

# -----------------------------------------------------------------------------------
# 1. Sentimento (funcionalidade principal)
# -----------------------------------------------------------------------------------
with aba_sent:
    u, ant = base.iloc[-1], base.iloc[-2]
    var = (u["Fechamento"] / ant["Fechamento"] - 1)
    ult = base.index[-1]
    sent_dia, sent_5 = u["Sent_Ponderado"], base["Sent_Ponderado"].tail(5).mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        cor = VERDE if var >= 0 else VERMELHO
        cartao(f"Ibovespa · fechamento em {ult:%d/%m/%Y}", br(u["Fechamento"], 0),
               f"{'▲' if var >= 0 else '▼'} {pct(abs(var), 2)} no dia", cor)
    with c2:
        cls = S.classe_sentimento(sent_dia)
        cor = VERDE if cls == "Positiva" else VERMELHO if cls == "Negativa" else CINZA
        cartao("Sentimento das notícias do pregão", f"{'+' if sent_dia > 0 else ''}{br(sent_dia, 2)}",
               f"{cls} · {int(u['N_Noticias'])} notícias · escala de −1 a +1", cor, barra=(sent_dia + 1) / 2)
    with c3:
        cls5 = S.classe_sentimento(sent_5)
        cor = VERDE if cls5 == "Positiva" else VERMELHO if cls5 == "Negativa" else CINZA
        cartao("Sentimento médio dos últimos 5 pregões", f"{'+' if sent_5 > 0 else ''}{br(sent_5, 2)}",
               f"{cls5} · tendência recente do noticiário", cor, barra=(sent_5 + 1) / 2)

    if fb:
        st.caption(f"Sentimento calculado pelo FinBERT-PT-BR ajustado, que acerta {pct(fb['acc_num'])} "
                   f"das notícias rotuladas por especialistas (validação cruzada em 5 divisões; "
                   f"o original acertava {pct(fb['acc_orig_num'])}).")

    # Gráfico: preço x sentimento
    janela = base.tail(120)
    fig, ax = plt.subplots(figsize=(10, 3.0), dpi=140)
    fig.patch.set_alpha(0)
    cores = [VERDE if s >= 0 else VERMELHO for s in janela["Sent_Ponderado"]]
    ax.bar(janela.index, janela["Sent_Ponderado"], color=cores, alpha=0.35, width=1.0, label="sentimento do dia")
    ax.plot(janela.index, janela["Sent_Ponderado"].rolling(5).mean(), color=AZUL, lw=1.6,
            label="média de 5 pregões")
    ax.axhline(0, color=CINZA, lw=0.6)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("sentimento", color=CINZA, fontsize=8)
    eixos_neutros(ax)
    ax2 = ax.twinx()
    ax2.plot(janela.index, janela["Fechamento"], color=CINZA, lw=1.2, label="Ibovespa")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: br(v, 0)))
    eixos_neutros(ax2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    linhas = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(linhas, [h.get_label() for h in linhas], fontsize=7, loc="upper left", frameon=False,
              labelcolor=CINZA)
    fig.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)

    # Manchetes
    st.subheader("Manchetes dos últimos 3 pregões")
    usadas, posteriores = S.noticias_dos_pregoes(noticias, base.index, 3)
    filtro = st.radio("Mostrar", ["Mais relevantes", "Positivas", "Negativas", "Neutras"], horizontal=True)
    lista = usadas.copy()
    lista["classe"] = lista["score"].apply(S.classe_sentimento)
    if filtro == "Mais relevantes":
        lista = lista.reindex(lista["score"].abs().sort_values(ascending=False).index)
    else:
        lista = lista[lista["classe"] == filtro[:-1]].sort_values("datahora_utc", ascending=False)
    n_pos = (usadas["score"] >= 0.15).sum()
    n_neg = (usadas["score"] <= -0.15).sum()
    st.caption(f"{len(usadas)} manchetes: {n_pos} positivas, {n_neg} negativas e "
               f"{len(usadas) - n_pos - n_neg} neutras.")
    if lista.empty:
        st.info("Nenhuma manchete nessa categoria.")
    for i in range(0, min(len(lista), 12), 3):
        cols = st.columns(3)
        for col, (_, n) in zip(cols, lista.iloc[i:i + 3].iterrows()):
            with col, st.container(border=True):
                url = str(n.get("url", "") or "")
                st.markdown(f"**[{n['titulo']}]({url})**" if url.startswith("http") else f"**{n['titulo']}**")
                st.caption(f"{n.get('veiculo', '')} · {n['datahora_brasilia']:%d/%m %H:%M}")
                cor = "green" if n["classe"] == "Positiva" else "red" if n["classe"] == "Negativa" else "gray"
                st.markdown(f":{cor}[{n['classe']} ({'+' if n['score'] > 0 else ''}{br(n['score'], 2)})]")
    if not posteriores.empty:
        with st.expander(f"Publicadas após o último fechamento ({len(posteriores)}): entram no próximo pregão"):
            for _, n in posteriores.sort_values("datahora_utc", ascending=False).head(15).iterrows():
                st.markdown(f"- {n['titulo']} · *{S.classe_sentimento(n['score'])}* "
                            f"({'+' if n['score'] > 0 else ''}{br(n['score'], 2)})")

# -----------------------------------------------------------------------------------
# 2. Analisar manchete (demonstração ao vivo)
# -----------------------------------------------------------------------------------
with aba_manchete:
    st.subheader("Teste o modelo com qualquer manchete")
    st.caption("O FinBERT-PT-BR ajustado classifica a manchete como positiva, negativa ou neutra para o mercado.")
    exemplos = ["Dólar volta a cair e vai a R$ 4,01",
                "Petrobras anuncia lucro recorde e dividendos acima do esperado",
                "Ibovespa recua com tensão fiscal e alta dos juros futuros",
                "Banco Central mantém a Selic em 10,5% ao ano"]
    escolha = st.selectbox("Exemplos", ["(digite a sua)"] + exemplos)
    texto = st.text_area("Manchete", value="" if escolha == "(digite a sua)" else escolha, height=80)
    if st.button("Analisar", type="primary") and texto.strip():
        with st.spinner("Carregando o modelo e analisando..."):
            r = S.analisar_manchete(carregar_finbert(info["nome"]), texto.strip())
        cor = VERDE if r["classe"] == "Positiva" else VERMELHO if r["classe"] == "Negativa" else CINZA
        c1, c2 = st.columns([1, 2])
        with c1:
            cartao("Classificação", r["classe"],
                   f"score {'+' if r['score'] > 0 else ''}{br(r['score'], 2)} (P(positivo) − P(negativo))", cor)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 1.9), dpi=140)
            fig.patch.set_alpha(0)
            nomes, vals = ["Positiva", "Negativa", "Neutra"], [r["positivo"], r["negativo"], r["neutro"]]
            ax.barh(nomes, vals, color=[VERDE, VERMELHO, CINZA])
            for y, v in enumerate(vals):
                ax.text(v + 0.01, y, pct(v), va="center", color=CINZA, fontsize=8)
            ax.set_xlim(0, 1.1)
            ax.invert_yaxis()
            eixos_neutros(ax)
            fig.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close(fig)
        st.caption("Atenção: manchetes que misturam movimentos opostos (ex.: \"Petrobras sobe e Vale cai\") "
                   "recebem um único score para a frase inteira; essa é uma limitação conhecida do modelo.")

# -----------------------------------------------------------------------------------
# 3. Sinais experimentais
# -----------------------------------------------------------------------------------
with aba_sinais:
    st.subheader("Sinais experimentais para o próximo pregão")
    st.warning("Estes sinais **não superaram o acaso com significância estatística** na validação "
               "walk-forward. Eles estão aqui para mostrar o funcionamento completo da solução e os seus "
               "limites. Não use para decidir investimentos.")
    cols = st.columns(2)
    for col, alvo in zip(cols, ("direcao", "risco")):
        cfg = S.SINAIS[alvo]
        with col:
            r, erro = sinal(alvo, st.session_state.versao)
            d = S.desempenho_sinal(tabelas, alvo)
            if r is None:
                st.info(f"**{cfg['titulo']}**: modelo ainda não salvo. Rode:\n\n`{cfg['comando']}`"
                        + (f"\n\nErro: {erro}" if erro else ""))
                continue
            p = r["prob"]
            classe = cfg["classe1"] if p >= 0.5 else cfg["classe0"]
            if alvo == "direcao":
                cor = VERDE if p >= 0.5 else VERMELHO
            else:
                cor = VERMELHO if p >= 0.5 else VERDE
            forca = abs(p - 0.5)
            intensidade = "sinal fraco" if forca < 0.05 else "sinal moderado" if forca < 0.10 else "sinal forte"
            cartao(f"{cfg['titulo']} ({r['proximo']:%d/%m/%Y})", classe,
                   f"P({cfg['classe1'].lower()}) = {pct(p)} · {intensidade}", cor, barra=p)
            if d:
                txt = (f"Desempenho medido fora da amostra (5 folds, 2024 a 2026): **{pct(d['acuracia'])}** "
                       f"de acerto, contra {pct(d['baseline'])} de um chute constante "
                       f"(p = {br(d['p_valor'], 3)}; acima do baseline em {d['folds_acima'] or '?'} folds).")
                if alvo == "risco" and d["persistencia"] is not None:
                    txt += f" Regra simples sem IA: {pct(d['persistencia'])}."
                st.markdown(txt)
            st.caption(f"Modelo: {cfg['chave']} · treinado com dados até {r['fim_treino']:%d/%m/%Y} · "
                       f"dados até {r['data_ref']:%d/%m/%Y}")

# -----------------------------------------------------------------------------------
# 4. Validação
# -----------------------------------------------------------------------------------
with aba_valid:
    st.subheader("1. Componente de linguagem: FinBERT-PT-BR")
    if fb:
        c1, c2, c3 = st.columns(3)
        c1.metric("Acurácia (ajustado)", pct(fb["acc_num"]), f"{br((fb['acc_num'] - fb['acc_orig_num']) * 100, 1)} p.p. vs original")
        c2.metric("ROC-AUC", fb["auc"].split(" ")[0].replace(".", ","))
        c3.metric("Baseline (chute)", pct(fb["baseline"]))
        st.caption("1.579 notícias rotuladas por pesquisadores da Unicamp (Carosia et al., 2022; "
                   "doi:10.25824/REDU/GFJHFK). Validação cruzada em 5 divisões: cada teste usa notícias "
                   "nunca vistas no treino.")
        st.table(tabelas["finbert_cv"])
    else:
        st.info("Rode `python finetune_finbert.py --arquivo Labeled.zip --cv 5` para gerar esta tabela.")

    st.subheader("2. Previsão de mercado: walk-forward em 5 folds")
    st.caption("Treino só com o passado; 5 períodos de teste consecutivos (abr/2024 a nov/2026); p-valor "
               "binomial contra o chute constante. Um modelo só supera o acaso com p < 0,05 e acima do "
               "baseline na maioria dos folds.")
    for alvo, titulo in (("direcao", "Direção (alta ou baixa) do próximo pregão"),
                         ("risco", "Risco (oscilação acima do normal) do próximo pregão")):
        t = tabelas.get(alvo)
        st.markdown(f"**{titulo}**")
        if t is None:
            st.info(f"Rode `python main_ensemble.py{' --alvo risco' if alvo == 'risco' else ''} --comparar`.")
            continue
        cols = [c for c in ["acuracia", "baseline", "ganho_pp", "p_valor", "auc", "f1_macro"] if c in t.columns]
        st.table(t[cols])
        f = tabelas.get(f"{alvo}_folds")
        if f is not None:
            with st.expander("Acurácia por fold"):
                st.table(f)
    if tabelas.get("risco_seletiva") is not None:
        with st.expander("Previsão seletiva do risco (só nos dias de maior confiança)"):
            st.table(tabelas["risco_seletiva"])
            st.caption("Foram cerca de 70 combinações testadas; com tantos testes, alguns p < 0,05 surgem "
                       "por acaso. Por isso nenhum resultado isolado é tratado como evidência.")

# -----------------------------------------------------------------------------------
# 5. Limitações e uso responsável
# -----------------------------------------------------------------------------------
with aba_limites:
    st.subheader("Análise crítica")
    st.markdown("""
**O que funciona**
- O FinBERT-PT-BR ajustado lê notícias financeiras brasileiras com cerca de 88% de acerto, contra 77%
  do modelo original, e corrigiu o viés pessimista (o original quase não reconhecia notícias positivas).

**O que não funciona, e por quê**
- **Notícia não vira previsão do preço de amanhã.** Com validação rigorosa, direção e risco ficaram
  entre 50% e 55%. Isso é coerente com a hipótese de mercados eficientes: o que está nas notícias
  tende a ser incorporado ao preço rapidamente.
- **Mudança de regime.** Modelos treinados em um período (ex.: alta) erram quando o mercado muda
  (o último fold teve só 43% de dias de alta).
- **Teto do alvo diário.** A oscilação de um único dia tem muito ruído próprio; mesmo conhecendo a
  volatilidade, não dá para acertar com folga se um dia específico ficará acima do normal.

**Limitações do componente de linguagem**
- Manchetes com movimentos opostos ("Petrobras sobe e BRF cai") recebem um score único.
- Contexto de mercado nem sempre é entendido (dólar em queda costuma ser bom para a bolsa).
- Base de treino de 2017 a 2019; vocabulário de eventos recentes pode diferir.

**Vieses e uso responsável**
- O noticiário financeiro tem viés de cobertura (mais notícias em dias de queda) e de fonte
  (portais específicos).
- Backtests não incluem custos de corretagem, impostos e slippage.
- Muitas configurações foram testadas; resultados isolados com p < 0,05 podem ser acaso.
- **Não é recomendação de investimento.** O app apoia a leitura do noticiário; a decisão é do usuário.
""")