"""
Late fusion: combina as probabilidades de saída dos dois ramos já treinados.

    Ramo temporal -> P_A(alta) --\
                                  >-- fusão --> P(alta) final
    Ramo de texto -> P_B(alta) --/

A fusão é ajustada no bloco "meta", que nenhum modelo base viu. Estratégias:
  media      média simples (0,5 e 0,5), sem nada a ajustar
  ponderada  pesos proporcionais à habilidade de cada ramo no bloco meta (AUC - 0,5)
  stacking   meta-modelo: regressão logística sobre os logits dos dois ramos, com
             pesos não negativos (um ramo com peso negativo seria um modelo "invertido"
             por causa de ruído no bloco meta; nesse caso ele é descartado)

Com blocos meta pequenos (~200 pregões), a média simples costuma ser a mais robusta.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LateFusion:
    def __init__(self, estrategia="media", limiar="fixo"):
        if estrategia not in ("media", "ponderada", "stacking"):
            raise ValueError("estrategia deve ser media, ponderada ou stacking")
        self.estrategia, self.modo_limiar = estrategia, limiar
        self.pesos, self.meta, self.limiar = (0.5, 0.5), None, 0.5
        self.estrategia_efetiva = estrategia

    def fit(self, pA, pB, y):
        if self.estrategia == "ponderada":
            habilidade = np.array([max(roc_auc_score(y, pA) - 0.5, 0.0),
                                   max(roc_auc_score(y, pB) - 0.5, 0.0)])
            self.pesos = tuple(habilidade / habilidade.sum()) if habilidade.sum() > 0 else (0.5, 0.5)
        elif self.estrategia == "stacking":
            Z = np.column_stack([_logit(pA), _logit(pB)])
            self.ativos = [0, 1]
            while self.ativos:
                self.meta = LogisticRegression(C=1.0, class_weight="balanced")
                self.meta.fit(Z[:, self.ativos], y)
                negativos = [a for a, c in zip(self.ativos, self.meta.coef_[0]) if c <= 0]
                if not negativos:
                    break
                self.ativos = [a for a in self.ativos if a not in negativos]
            if not self.ativos:                 # nenhum ramo útil no bloco meta: média simples
                self.estrategia_efetiva = "media"
        if self.modo_limiar == "meta":
            # limiar que maximiza a acurácia balanceada no bloco meta (nunca no teste)
            p = self.predict_proba(pA, pB)
            candidatos = np.quantile(p, np.linspace(0.2, 0.8, 61))
            self.limiar = float(max(candidatos, key=lambda l: balanced_accuracy_score(y, p >= l)))
        return self

    def predict_proba(self, pA, pB):
        if self.estrategia_efetiva == "stacking":
            Z = np.column_stack([_logit(pA), _logit(pB)])[:, self.ativos]
            return self.meta.predict_proba(Z)[:, 1]
        return self.pesos[0] * pA + self.pesos[1] * pB

    def descrever(self):
        if self.estrategia_efetiva == "stacking":
            coef = dict(zip(self.ativos, self.meta.coef_[0]))
            return (f"stacking (coef. temporal {coef.get(0, 0.0):+.3f}, texto {coef.get(1, 0.0):+.3f}), "
                    f"limiar {self.limiar:.3f}")
        if self.estrategia == "stacking":
            return f"stacking sem ramo útil -> média simples, limiar {self.limiar:.3f}"
        return (f"{self.estrategia} (pesos temporal {self.pesos[0]:.2f}, texto {self.pesos[1]:.2f}), "
                f"limiar {self.limiar:.3f}")
