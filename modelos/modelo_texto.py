"""
Modelo 2: ramo de texto (FinBERT + calibração).

O FinBERT-PT-BR (pré-treinado, não é retreinado) já transformou cada manchete em
probabilidades de sentimento em gerar_base.py, e o sentimento foi agregado por pregão.
Este módulo aprende como esse sentimento se traduz em P(alta) no pregão seguinte,
com uma regressão logística regularizada (C escolhido na validação).

Usar só o score bruto como probabilidade não funciona: o noticiário financeiro tem
viés negativo constante, e a relação entre tom e retorno precisa ser calibrada.

Entradas (dados_modelo.FEATURES_TEXTO): sentimento ponderado do dia e sua média de
5 pregões. Com só duas features, os coeficientes ficam interpretáveis: um peso
positivo significa "notícias mais otimistas -> maior chance de alta".
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

GRADE_C = (0.001, 0.01, 0.1, 1.0)


class ModeloTexto:
    nome = "FinBERT + calibração"

    def __init__(self, seed=42, grade=GRADE_C):
        self.seed, self.grade = seed, grade

    def fit(self, Xtr, ytr, Xva, yva):
        melhor = (-np.inf, None)
        for C in self.grade:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
            auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
            if auc > melhor[0]:
                melhor = (auc, m)
        self.auc_val, self.modelo = melhor
        self.hp = {"C": self.modelo.C}
        return self

    def predict_proba(self, X):
        return self.modelo.predict_proba(X)[:, 1]

    def coeficientes(self, nomes):
        return {n: round(float(c), 4) for n, c in zip(nomes, self.modelo.coef_[0])}

    def salvar(self, caminho, extras=None):
        joblib.dump({"modelo": self.modelo, "hp": self.hp, **(extras or {})}, caminho)


def criar(seed=42, **ajustes):
    return ModeloTexto(seed, **ajustes)