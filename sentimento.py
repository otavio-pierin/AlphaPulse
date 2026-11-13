"""
Análise de sentimento das manchetes com FinBERT (Hugging Face, gratuito).

Modelo padrão: lucas-leme/FinBERT-PT-BR, um BERT treinado com textos financeiros em
português que classifica cada texto em positivo, negativo ou neutro.
Score contínuo = P(positivo) - P(negativo), no intervalo [-1, 1].
"""

import numpy as np
import pandas as pd


class FinBERT:
    def __init__(self, modelo="lucas-leme/FinBERT-PT-BR", batch=32, max_tokens=128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.nome = torch, modelo
        self.batch, self.max_tokens = batch, max_tokens
        self.disp = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sentimento] Carregando {modelo} em {self.disp}...")
        self.tok = AutoTokenizer.from_pretrained(modelo)
        try:                                   # safetensors: seguro e independe da versão do torch
            self.rede = AutoModelForSequenceClassification.from_pretrained(modelo, use_safetensors=True)
        except (OSError, ValueError):
            self.rede = AutoModelForSequenceClassification.from_pretrained(modelo)
        self.rede.to(self.disp).eval()

        # Cada modelo ordena os rótulos de um jeito: localiza pelo nome
        rotulos = {i: str(r).lower() for i, r in self.rede.config.id2label.items()}
        achar = lambda chave: next((i for i, r in rotulos.items() if chave in r), None)
        self.i_pos, self.i_neg, self.i_neu = achar("pos"), achar("neg"), achar("neu")
        if self.i_pos is None or self.i_neg is None:
            raise ValueError(f"Rótulos inesperados em {modelo}: {rotulos}")

    def probabilidades(self, textos):
        """Matriz (n, 3) com P(positivo), P(negativo), P(neutro)."""
        saida = []
        with self.torch.no_grad():
            for i in range(0, len(textos), self.batch):
                enc = self.tok([str(t) for t in textos[i:i + self.batch]], padding=True,
                               truncation=True, max_length=self.max_tokens,
                               return_tensors="pt").to(self.disp)
                p = self.torch.softmax(self.rede(**enc).logits, dim=-1).cpu().numpy()
                neu = p[:, self.i_neu] if self.i_neu is not None else np.zeros(len(p))
                saida.append(np.column_stack([p[:, self.i_pos], p[:, self.i_neg], neu]))
        return np.vstack(saida) if saida else np.zeros((0, 3))

    def score(self, textos):
        p = self.probabilidades(textos)
        return p[:, 0] - p[:, 1]


def pontuar_pendentes(banco, cfg, bloco=512):
    """
    Pontua só as manchetes ainda sem score deste modelo e grava em blocos:
    se a execução for interrompida, o que já foi pontuado fica salvo.
    """
    f = cfg["finbert"]
    pend = banco.sem_sentimento(f["modelo"])
    if pend.empty:
        print("[sentimento] Todas as manchetes já têm score.")
        return 0

    modelo = FinBERT(f["modelo"], f["batch"], f["max_tokens"])
    textos = (pend["titulo"] + ". " + pend["resumo"].fillna("") if f["usar_resumo"]
              else pend["titulo"]).tolist()
    print(f"[sentimento] Pontuando {len(textos)} manchetes...")
    for i in range(0, len(textos), bloco):
        p = modelo.probabilidades(textos[i:i + bloco])
        ids = pend["id"].iloc[i:i + bloco].tolist()
        banco.gravar_sentimento([(float(a), float(b), float(c), float(a - b), f["modelo"], j)
                                 for (a, b, c), j in zip(p, ids)])
        print(f"   {min(i + bloco, len(textos))}/{len(textos)}")
    return len(textos)


def agregar_por_pregao(noticias, datas_pregao):
    """
    Agrega o sentimento por pregão. Cada notícia vai para o primeiro pregão igual ou
    posterior à sua data efetiva (fins de semana e feriados contam para o próximo
    dia útil). Dias sem notícia ficam com score 0 e Sem_Noticia = 1.
    """
    n = noticias.dropna(subset=["score"]).sort_values("data_efetiva")
    pregoes = pd.DataFrame({"Data": pd.DatetimeIndex(datas_pregao).as_unit("ns")})
    m = pd.merge_asof(n, pregoes, left_on="data_efetiva", right_on="Data", direction="forward")
    m = m.dropna(subset=["Data"])
    m["confianca"] = 1 - m["prob_neu"]
    m["score_x_conf"] = m["score"] * m["confianca"]
    m["positiva"] = (m["score"] > 0.15).astype(float)
    m["negativa"] = (m["score"] < -0.15).astype(float)

    g = m.groupby("Data")
    diario = pd.DataFrame({
        "Sent_Medio": g["score"].mean(),
        # média ponderada pela confiança: manchetes neutras ou ambíguas pesam menos
        "Sent_Ponderado": g["score_x_conf"].sum() / g["confianca"].sum().clip(lower=1e-9),
        "Sent_Desvio": g["score"].std(),
        "Prop_Positivas": g["positiva"].mean(),
        "Prop_Negativas": g["negativa"].mean(),
        "N_Noticias": g.size(),
    })
    diario = diario.reindex(pregoes["Data"])
    diario["Sem_Noticia"] = diario["N_Noticias"].isna().astype(int)
    diario = diario.fillna({"Sent_Medio": 0.0, "Sent_Ponderado": 0.0, "Sent_Desvio": 0.0,
                            "Prop_Positivas": 0.0, "Prop_Negativas": 0.0, "N_Noticias": 0})
    diario.index.name = "Data"
    return diario