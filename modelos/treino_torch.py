"""
Treino compartilhado pelas redes PyTorch (MLP, LSTM, Bi-LSTM, CNN).

Todas recebem a mesma interface de um classificador binário:
    modelo.fit(X_treino, y_treino, X_val, y_val)
    modelo.predict_proba(X) -> P(alta)
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixar_sementes(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ClassificadorTorch:
    """
    Envolve um nn.Module que devolve um logit por amostra.
    Perda BCE com pos_weight (compensa desbalanceamento), Adam com weight decay,
    early stopping pela perda de validação e restauração dos melhores pesos.
    """

    def __init__(self, nome, construtor, hiperparametros, seed=42):
        self.nome, self.construtor = nome, construtor
        self.hp = dict(hiperparametros)
        self.seed = seed
        self.rede = None
        self.historico = {"loss": [], "val_loss": []}

    def fit(self, Xtr, ytr, Xva, yva, verbose=False):
        fixar_sementes(self.seed)
        disp = dispositivo()
        self.rede = self.construtor(Xtr.shape, **self.hp.get("arquitetura", {})).to(disp)

        p1 = float(np.mean(ytr))
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - p1) / p1, device=disp))
        opt = torch.optim.Adam(self.rede.parameters(), lr=self.hp["lr"],
                               weight_decay=self.hp.get("weight_decay", 1e-4))
        gerador = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                            batch_size=self.hp["batch_size"], shuffle=True, generator=gerador)
        Xva_t = torch.from_numpy(Xva).to(disp)
        yva_t = torch.from_numpy(yva).to(disp)

        melhor, melhor_estado, sem_melhora = float("inf"), None, 0
        inicio = time.time()
        for ep in range(self.hp["epochs"]):
            self.rede.train()
            soma, total = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(disp), yb.to(disp)
                opt.zero_grad()
                perda = crit(self.rede(xb), yb)
                perda.backward()
                nn.utils.clip_grad_norm_(self.rede.parameters(), 1.0)
                opt.step()
                soma += perda.item() * len(yb)
                total += len(yb)
            self.rede.eval()
            with torch.no_grad():
                val = crit(self.rede(Xva_t), yva_t).item()
            self.historico["loss"].append(soma / total)
            self.historico["val_loss"].append(val)
            if verbose:
                print(f"      época {ep + 1:3d} loss {soma / total:.4f} val_loss {val:.4f}")
            if val < melhor - 1e-5:
                melhor, sem_melhora = val, 0
                melhor_estado = copy.deepcopy(self.rede.state_dict())
            else:
                sem_melhora += 1
                if sem_melhora >= self.hp["patience"]:
                    break
        if melhor_estado is not None:
            self.rede.load_state_dict(melhor_estado)
        self.rede.eval()
        self.epocas = ep + 1
        self.tempo = time.time() - inicio
        return self

    def predict_proba(self, X, batch=512):
        disp = next(self.rede.parameters()).device
        saida = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(disp)
                saida.append(torch.sigmoid(self.rede(xb)).cpu().numpy())
        return np.concatenate(saida)

    def salvar(self, caminho, extras=None):
        torch.save({"nome": self.nome, "hiperparametros": self.hp, "seed": self.seed,
                    "state_dict": {k: v.cpu() for k, v in self.rede.state_dict().items()},
                    **(extras or {})}, caminho)

    def n_parametros(self):
        return sum(p.numel() for p in self.rede.parameters() if p.requires_grad)
