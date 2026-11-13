"""
Modelo 1 (opção): Support Vector Machine (técnica clássica, scikit-learn).
O hiperparâmetro C é escolhido na validação pela ROC-AUC.
As probabilidades vêm de uma calibração sigmoide (Platt) feita com validação cruzada
interna, a forma recomendada pelo scikit-learn a partir da versão 1.9.
"""

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

TIPO = "tabular"
GRADE_C = (0.1, 0.3, 1.0, 3.0, 10.0)


class ModeloSVM:
    nome = "SVM"

    def __init__(self, seed=42, grade=GRADE_C, kernel="rbf"):
        self.seed, self.grade, self.kernel = seed, grade, kernel

    def fit(self, Xtr, ytr, Xva, yva):
        melhor = (-np.inf, None)
        for C in self.grade:
            svc = SVC(C=C, kernel=self.kernel, gamma="scale", class_weight="balanced")
            m = CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False).fit(Xtr, ytr)
            auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
            if auc > melhor[0]:
                melhor = (auc, m)
        self.auc_val, self.modelo = melhor
        self.hp = {"C": self.modelo.estimator.C, "kernel": self.kernel}
        return self

    def predict_proba(self, X):
        return self.modelo.predict_proba(X)[:, 1]

    def salvar(self, caminho, extras=None):
        joblib.dump({"modelo": self.modelo, "hp": self.hp, **(extras or {})}, caminho)


def criar(seed=42, **ajustes):
    return ModeloSVM(seed, **ajustes)