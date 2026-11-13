"""
Publica o FinBERT-PT-BR ajustado no Hugging Face Hub (gratuito), para não versionar
o arquivo de 415 MB no GitHub.

Antes (uma vez):
    pip install huggingface_hub
    Crie um token com permissão de escrita em https://huggingface.co/settings/tokens

Uso:
    python publicar_modelo.py --repo SEU_USUARIO/finbert-pt-br-ft

Depois, em config.py:  "modelo": "SEU_USUARIO/finbert-pt-br-ft"
O código baixa o modelo automaticamente na primeira execução.
"""

import argparse
import os

CARTAO = """---
language: pt
license: cc-by-nc-4.0
base_model: lucas-leme/FinBERT-PT-BR
tags: [finance, sentiment-analysis, portuguese, brazil]
---

# FinBERT-PT-BR ajustado (AlphaPulse)

Fine-tuning do [lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR) para
sentimento de notícias financeiras brasileiras (positiva, negativa, neutra), feito no projeto
acadêmico AlphaPulse (SIN 323, UFV).

**Dados de treino:** notícias rotuladas de Carosia, Silva e Coelho (2022), Repositório de Dados de
Pesquisa da Unicamp, doi:10.25824/REDU/GFJHFK, licença CC BY-NC 4.0. Por isso este modelo também é
disponibilizado apenas para uso não comercial.

**Avaliação (validação cruzada em 5 divisões, notícias nunca vistas no treino):**
acurácia de 88,4% ± 1,1% em positivas x negativas, contra 77,1% do modelo original;
ROC-AUC de 0,958.

**Limitações:** manchetes com movimentos opostos recebem um único score; base de treino de 2017 a 2019.
Não é recomendação de investimento.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ex.: otavio-pierin/finbert-pt-br-ft")
    ap.add_argument("--pasta", default="modelos_finbert/finbert-pt-br-ft")
    ap.add_argument("--privado", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, login

    if not os.path.exists(os.path.join(a.pasta, "model.safetensors")):
        raise SystemExit(f"Modelo não encontrado em {a.pasta}. Rode finetune_finbert.py antes.")
    with open(os.path.join(a.pasta, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARTAO)

    login()                                    # pede o token na primeira vez
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=a.privado, exist_ok=True)
    api.upload_folder(folder_path=a.pasta, repo_id=a.repo, repo_type="model",
                      commit_message="FinBERT-PT-BR ajustado (AlphaPulse)")
    print(f"\n[OK] Publicado em https://huggingface.co/{a.repo}")
    print(f'Em config.py, use: "modelo": "{a.repo}"')


if __name__ == "__main__":
    main()