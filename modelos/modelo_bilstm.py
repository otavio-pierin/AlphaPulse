"""
Modelo 1 (opção): Bidirectional LSTM.
Lê a janela nos dois sentidos; a representação final concatena os dois estados.
"""

from .modelo_lstm import LSTMClassificador
from .treino_torch import ClassificadorTorch

TIPO = "sequencia"
HIPERPARAMETROS = {"lr": 1e-3, "batch_size": 32, "epochs": 150, "patience": 15,
                   "weight_decay": 1e-4,
                   "arquitetura": {"oculto": 32, "camadas": 1, "dropout": 0.3, "bidirecional": True}}


def criar(seed=42, **ajustes):
    return ClassificadorTorch("Bi-LSTM", LSTMClassificador, {**HIPERPARAMETROS, **ajustes}, seed)
