"""
Modelo 1 (opção): Long Short-Term Memory.
Entrada sequencial: janela dos últimos N pregões de indicadores de preço.
"""

import torch
import torch.nn as nn

from .treino_torch import ClassificadorTorch

TIPO = "sequencia"
HIPERPARAMETROS = {"lr": 1e-3, "batch_size": 32, "epochs": 150, "patience": 15,
                   "weight_decay": 1e-4, "arquitetura": {"oculto": 32, "camadas": 1, "dropout": 0.3}}


class LSTMClassificador(nn.Module):
    def __init__(self, forma, oculto=32, camadas=1, dropout=0.3, bidirecional=False):
        super().__init__()
        self.lstm = nn.LSTM(forma[-1], oculto, num_layers=camadas, batch_first=True,
                            dropout=dropout if camadas > 1 else 0.0, bidirectional=bidirecional)
        saida = oculto * (2 if bidirecional else 1)
        self.cabeca = nn.Sequential(nn.Dropout(dropout), nn.Linear(saida, 1))
        self.bidirecional = bidirecional

    def forward(self, x):                      # x: (amostras, janela, features)
        _, (h, _) = self.lstm(x)
        if self.bidirecional:                  # última camada: sentido direto e reverso
            h = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h = h[-1]
        return self.cabeca(h).squeeze(-1)


def criar(seed=42, **ajustes):
    return ClassificadorTorch("LSTM", LSTMClassificador, {**HIPERPARAMETROS, **ajustes}, seed)
