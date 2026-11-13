"""
Modelo 1 (opção): Multilayer Perceptron.
Entrada tabular: indicadores do dia t + retornos dos 5 dias anteriores.
"""

import torch.nn as nn

from .treino_torch import ClassificadorTorch

TIPO = "tabular"
HIPERPARAMETROS = {"lr": 1e-3, "batch_size": 32, "epochs": 200, "patience": 20,
                   "weight_decay": 1e-4, "arquitetura": {"camadas": (64, 32), "dropout": 0.3}}


class MLP(nn.Module):
    def __init__(self, forma, camadas=(64, 32), dropout=0.3):
        super().__init__()
        blocos, entrada = [], forma[-1]
        for n in camadas:
            blocos += [nn.Linear(entrada, n), nn.BatchNorm1d(n), nn.ReLU(), nn.Dropout(dropout)]
            entrada = n
        self.rede = nn.Sequential(*blocos, nn.Linear(entrada, 1))

    def forward(self, x):                      # x: (amostras, features)
        return self.rede(x).squeeze(-1)


def criar(seed=42, **ajustes):
    return ClassificadorTorch("MLP", MLP, {**HIPERPARAMETROS, **ajustes}, seed)
