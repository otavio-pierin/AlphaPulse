"""
Modelo 1 (opção): Convolutional Neural Network 1D.
Convoluções ao longo do tempo extraem padrões locais; Global Average Pooling resume a janela.
"""

import torch.nn as nn

from .treino_torch import ClassificadorTorch

TIPO = "sequencia"
HIPERPARAMETROS = {"lr": 1e-3, "batch_size": 32, "epochs": 150, "patience": 15,
                   "weight_decay": 1e-4, "arquitetura": {"filtros": 32, "kernel": 3, "dropout": 0.3}}


class CNN1D(nn.Module):
    def __init__(self, forma, filtros=32, kernel=3, dropout=0.3):
        super().__init__()
        f = forma[-1]
        self.conv = nn.Sequential(
            nn.Conv1d(f, filtros, kernel, padding=kernel // 2), nn.BatchNorm1d(filtros), nn.ReLU(),
            nn.Conv1d(filtros, filtros, kernel, padding=kernel // 2), nn.BatchNorm1d(filtros), nn.ReLU(),
        )
        self.cabeca = nn.Sequential(nn.Dropout(dropout), nn.Linear(filtros, 1))

    def forward(self, x):                      # x: (amostras, janela, features)
        h = self.conv(x.transpose(1, 2))       # Conv1d espera (amostras, canais, tempo)
        return self.cabeca(h.mean(dim=-1)).squeeze(-1)


def criar(seed=42, **ajustes):
    return ClassificadorTorch("CNN", CNN1D, {**HIPERPARAMETROS, **ajustes}, seed)
