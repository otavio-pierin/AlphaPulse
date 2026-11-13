"""
Modelo temporal: Transformer encoder para séries temporais.

Cada pregão da janela vira um "token": suas features são projetadas para d_model
dimensões e somadas a uma codificação de posição aprendida. A autoatenção permite que
cada dia da janela pondere a relevância dos outros dias. A saída é a média dos tokens.

Usado de duas formas em main_ensemble.py:
  late fusion  -> recebe só a janela de preço; sua P(alta) é combinada com o ramo de texto
  early fusion -> recebe preço + sentimento concatenados em cada dia da janela

O modelo é pequeno de propósito (~25 mil parâmetros): Transformers precisam de muitos
dados, e aqui há cerca de 1.200 a 1.800 janelas de treino por fold.
"""

import torch
import torch.nn as nn

from .treino_torch import ClassificadorTorch

TIPO = "sequencia"
HIPERPARAMETROS = {"lr": 5e-4, "batch_size": 32, "epochs": 150, "patience": 15,
                   "weight_decay": 1e-3,
                   "arquitetura": {"d_model": 32, "cabecas": 4, "camadas": 2, "ff": 64, "dropout": 0.2}}


class TransformerClassificador(nn.Module):
    def __init__(self, forma, d_model=32, cabecas=4, camadas=2, ff=64, dropout=0.2):
        super().__init__()
        janela, n_features = forma[1], forma[2]
        self.entrada = nn.Linear(n_features, d_model)
        self.posicao = nn.Parameter(torch.zeros(1, janela, d_model))
        nn.init.normal_(self.posicao, std=0.02)
        camada = nn.TransformerEncoderLayer(d_model, cabecas, ff, dropout,
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(camada, camadas, enable_nested_tensor=False)
        self.norma = nn.LayerNorm(d_model)
        self.cabeca = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):                      # x: (amostras, janela, features)
        h = self.encoder(self.entrada(x) + self.posicao)
        return self.cabeca(self.norma(h.mean(dim=1))).squeeze(-1)


def criar(seed=42, **ajustes):
    return ClassificadorTorch("Transformer", TransformerClassificador,
                              {**HIPERPARAMETROS, **ajustes}, seed)