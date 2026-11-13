"""
Fábrica dos modelos do ramo temporal (Modelo 1) e do ramo de texto (Modelo 2).
As redes PyTorch são importadas só quando escolhidas, para que SVM e regressão
funcionem mesmo sem o PyTorch instalado.
"""

import importlib

TEMPORAIS = {
    "lstm": "modelo_lstm",
    "bilstm": "modelo_bilstm",
    "cnn": "modelo_cnn",
    "transformer": "modelo_transformer",
    "mlp": "modelo_mlp",
    "svm": "modelo_svm",
    "logistica": "modelo_regressao",
}


def modulo_temporal(nome):
    if nome not in TEMPORAIS:
        raise ValueError(f"Modelo '{nome}' desconhecido. Opções: {', '.join(TEMPORAIS)}")
    return importlib.import_module(f".{TEMPORAIS[nome]}", __name__)


def criar_temporal(nome, seed=42):
    mod = modulo_temporal(nome)
    return mod.criar(seed), mod.TIPO


def criar_texto(seed=42):
    from .modelo_texto import criar
    return criar(seed)