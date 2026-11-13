"""
Modelo 1 (opção): regressão logística (técnica estatística clássica, baseline).
Serve de referência: se uma rede profunda não superá-la, a complexidade não se justifica.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TIPO = "tabular"
GRADE_C = (0.001, 0.01, 0.1, 1.0, 10.0)


class RegressaoLogistica:
    nome = "Regressão logística"

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

    def salvar(self, caminho, extras=None):
        joblib.dump({"modelo": self.modelo, "hp": self.hp, **(extras or {})}, caminho)


def criar(seed=42, **ajustes):
    return RegressaoLogistica(seed, **ajustes)
