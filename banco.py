"""
Armazenamento em SQLite (biblioteca padrão do Python).

Por que SQLite em vez de CSV: gravações são transações atômicas (um arquivo nunca
fica pela metade, mesmo se o processo for interrompido), a chave primária impede
duplicatas e várias execuções podem ler e gravar ao mesmo tempo com segurança.
"""

import os
import sqlite3

import pandas as pd

ESQUEMA = """
CREATE TABLE IF NOT EXISTS noticias (
    id            TEXT PRIMARY KEY,    -- hash do título normalizado
    fonte         TEXT NOT NULL,       -- google_news | wordpress
    veiculo       TEXT,
    titulo        TEXT NOT NULL,
    resumo        TEXT,
    url           TEXT,
    datahora_utc  TEXT NOT NULL,       -- ISO 8601
    data_efetiva  TEXT NOT NULL,       -- AAAA-MM-DD, já com o corte de horário aplicado
    prob_pos      REAL,
    prob_neg      REAL,
    prob_neu      REAL,
    score         REAL,                -- P(positivo) - P(negativo)
    modelo        TEXT
);
CREATE INDEX IF NOT EXISTS idx_noticias_data ON noticias(data_efetiva);
CREATE TABLE IF NOT EXISTS janelas (
    chave TEXT PRIMARY KEY,            -- fonte|consulta|inicio|fim
    n     INTEGER,
    feito TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class Banco:
    def __init__(self, caminho):
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
        self.con = sqlite3.connect(caminho, timeout=60)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(ESQUEMA)

    # --- checkpoints de coleta ---
    def janela_feita(self, chave):
        return self.con.execute("SELECT 1 FROM janelas WHERE chave=?", (chave,)).fetchone() is not None

    def marcar_janela(self, chave, n):
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO janelas(chave, n) VALUES (?, ?)", (chave, n))

    # --- notícias ---
    def inserir_noticias(self, registros):
        """Ignora notícias já existentes (mesmo título). Retorna quantas eram novas."""
        if not registros:
            return 0
        antes = self.total()
        with self.con:
            self.con.executemany(
                "INSERT OR IGNORE INTO noticias(id, fonte, veiculo, titulo, resumo, url, "
                "datahora_utc, data_efetiva) VALUES (:id, :fonte, :veiculo, :titulo, :resumo, "
                ":url, :datahora_utc, :data_efetiva)", registros)
        return self.total() - antes

    def total(self):
        return self.con.execute("SELECT COUNT(*) FROM noticias").fetchone()[0]

    def sem_sentimento(self, modelo):
        return pd.read_sql_query(
            "SELECT id, titulo, resumo FROM noticias WHERE score IS NULL OR modelo IS NOT ? "
            "ORDER BY datahora_utc", self.con, params=(modelo,))

    def gravar_sentimento(self, linhas):
        """linhas: lista de (prob_pos, prob_neg, prob_neu, score, modelo, id)."""
        with self.con:
            self.con.executemany(
                "UPDATE noticias SET prob_pos=?, prob_neg=?, prob_neu=?, score=?, modelo=? "
                "WHERE id=?", linhas)

    def ler_noticias(self):
        df = pd.read_sql_query("SELECT * FROM noticias ORDER BY datahora_utc", self.con)
        df["datahora_utc"] = pd.to_datetime(df["datahora_utc"], utc=True, format="ISO8601")
        df["data_efetiva"] = pd.to_datetime(df["data_efetiva"]).dt.as_unit("ns")
        return df

    def fechar(self):
        self.con.close()