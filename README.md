# AlphaPulse · Termômetro de sentimento das notícias do Ibovespa

Prova de Conceito (PoC) do projeto prático de **SIN 323 – Inteligência Artificial** (UFV – Campus Rio Paranaíba, 2026/2).

> Projeto acadêmico. **Não é recomendação de investimento.**

## O problema

Quem acompanha o Ibovespa precisa ler dezenas de manchetes por dia, em vários portais, para saber se o noticiário está otimista ou pessimista. A leitura é lenta, subjetiva e difícil de comparar entre dias. Modelos de sentimento genéricos (em inglês) erram o contexto do mercado brasileiro: por exemplo, "dólar cai" costuma ser bom para a bolsa.

## O que o AlphaPulse faz

1. **Coleta** manchetes sobre o Ibovespa (Google News RSS e API pública do Suno Notícias) e os preços do índice (Yahoo Finance).
2. **Classifica** cada manchete como positiva, negativa ou neutra para o mercado, com o **FinBERT-PT-BR ajustado** (fine-tuning).
3. **Agrega** o sentimento por pregão e mostra o termômetro, as manchetes que o compõem e um campo para analisar qualquer manchete digitada.
4. **Experimental:** testa se o sentimento ajuda a prever direção e risco do próximo pregão. O resultado foi próximo do acaso (50–55%) e é exibido como tal.

### Papel da IA

| | |
|---|---|
| Tarefa | Classificação de sentimento de texto (3 classes) |
| Entrada | Título da manchete em português |
| Saída | P(positivo), P(negativo), P(neutro) e score = P(pos) − P(neg) |
| Modelo | [FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR) com fine-tuning nas notícias rotuladas da Unicamp |
| Sem a IA | O produto não existe: não há como medir o sentimento de milhares de manchetes manualmente |

## Resultados

Fine-tuning avaliado com validação cruzada de 5 divisões (teste nunca visto no treino), em `resultados/finetune_finbert_cv.csv`:

| Modelo | Acurácia (pos × neg) | F1 macro |
|---|---|---|
| Baseline (classe majoritária) | 57,5% | – |
| FinBERT original | 77,1% ± 2,8 | 0,770 |
| **FinBERT ajustado** | **88,4% ± 1,1** | **0,881** |

Previsão de preço (walk-forward em 5 folds, `resultados/metricas_comparacao_*.csv`): sem superar o acaso com significância estatística, em linha com a hipótese de mercados eficientes. Muitas configurações foram testadas, então p < 0,05 isolados podem ser acaso.

**Validação em manchetes reais do produto:** os 88% vêm de notícias de 2017–2019. Para medir o desempenho nas manchetes que o app realmente exibe, use os scripts da seção [Validação em manchetes recentes](#validação-em-manchetes-recentes).

## Requisitos

- Python 3.10 ou superior
- Internet (coleta de notícias, preços e download do modelo no Hugging Face)
- GPU é opcional: o app funciona em CPU, mas o treino é bem mais rápido com GPU

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar a demonstração

```bash
python gerar_base.py             # 1) coleta preços e notícias e pontua o sentimento
streamlit run app.py             # 2) abre o app no navegador
```

- `gerar_base.py` cria a pasta `dados/` (ela não vem no repositório). Na primeira vez **pode demorar bastante** por causa da coleta histórica desde 2018. Ele guarda o progresso em um banco SQLite e **retoma de onde parou** se for interrompido.
- O modelo ajustado é baixado automaticamente do Hugging Face (`otaviopierin/finbert-pt-br-ft`, definido em `config.py`) na primeira execução.
- No app, o botão **Atualizar preços e notícias** busca o que for novo.
- Para a demonstração, **gere a base antes** e confira se o app abre sem depender da internet.

Abas do app: Sentimento · Analisar manchete · Sinais experimentais · Validação · Limitações.

### Aba "Sinais experimentais"

Ela usa modelos salvos em `resultados/`. Se aparecer "modelo ainda não salvo", gere-os:

```bash
python main_ensemble.py --temporal transformer --modo early     # direção
python main_ensemble.py --alvo risco --temporal lstm            # risco
```

## Reproduzir as validações

```bash
# Fine-tuning + avaliação do FinBERT (precisa do Labeled.zip, ver "Dados")
python finetune_finbert.py --arquivo Labeled.zip --cv 5
# Avaliação do FinBERT original
python avaliar_finbert.py --arquivo Labeled.zip
# Previsão de preço e risco (walk-forward)
python main_ensemble.py --comparar
python main_ensemble.py --alvo risco --comparar
# Publicar o modelo ajustado no Hugging Face (opcional)
python publicar_modelo.py
```

## Validação em manchetes recentes

Mede se o modelo continua bom nas manchetes reais do app (outras fontes e outros anos).

```bash
python amostrar_manchetes.py
# gera dados/validacao_manchetes_A.csv e dados/validacao_manchetes_B.csv
```

1. Duas pessoas do grupo preenchem a coluna `rotulo` de cada arquivo, **cada uma no seu e sem conversar**, com `pos`, `neg` ou `neu`.
   - `pos`: boa para o mercado/Ibovespa · `neg`: ruim · `neu`: sem efeito claro.
   - Julguem o efeito sobre a **bolsa**, não o tom da frase.
2. Depois:

```bash
python validar_manchetes_recentes.py dados/validacao_manchetes_A.csv dados/validacao_manchetes_B.csv
```

Saídas em `resultados/`: métricas (acurácia com intervalo de confiança, F1, kappa entre anotadores), acurácia por ano, lista de erros (úteis para a análise crítica) e matriz de confusão.

## Estrutura

| Arquivo | Função |
|---|---|
| `app.py`, `servico_app.py` | Interface Streamlit e lógica da aplicação (testável sem interface) |
| `gerar_base.py`, `noticias.py`, `pregoes.py`, `banco.py` | Coleta de notícias e preços, armazenamento em SQLite |
| `sentimento.py` | FinBERT: pontuação e agregação por pregão |
| `finetune_finbert.py`, `avaliar_finbert.py`, `publicar_modelo.py` | Ajuste fino, avaliação e publicação do modelo |
| `main_ensemble.py`, `ensemble.py`, `dados_modelo.py`, `avaliacao.py`, `modelos/` | Experimentos de previsão de direção e risco |
| `amostrar_manchetes.py`, `validar_manchetes_recentes.py` | Validação em manchetes reais rotuladas pelo grupo |
| `config.py` | Configuração central (ativo, período, fontes, modelo) |
| `resultados/` | Métricas, gráficos e modelos salvos |

## Limitações e uso responsável

- Manchetes com movimentos opostos ("Petrobras sobe e BRF cai") recebem um único score.
- O contexto de mercado nem sempre é entendido, e a base de treino é de 2017–2019.
- O noticiário tem viés de cobertura e de fonte (poucos portais).
- Notícia não vira previsão do preço de amanhã: não use os sinais experimentais para operar.
- Backtests não incluem corretagem, impostos nem *slippage*.

## Dados, licenças e referências

- **Notícias rotuladas:** Carosia, Silva e Coelho (2022), Repositório de Dados de Pesquisa da Unicamp, [doi:10.25824/REDU/GFJHFK](https://doi.org/10.25824/REDU/GFJHFK), licença CC BY-NC 4.0 (uso não comercial). Baixe o `Labeled.zip` do repositório; ele não está versionado.
- **Modelo base:** [lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR).
- **Notícias:** Google News (RSS) e Suno Notícias (API pública do WordPress). **Preços:** Yahoo Finance via `yfinance`.

## Autores

Lucas Brasil Oliveira, [nomes dos demais integrantes] · Sistemas de Informação, UFV – Campus Rio Paranaíba · Profa. Dra. Larissa Ferreira Rodrigues Moreira
