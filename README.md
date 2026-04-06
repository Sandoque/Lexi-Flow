# LexiFlow

Plataforma de Classificacao Textual Inteligente.

## Visao Geral

LexiFlow e uma aplicacao Flask com arquitetura profissional para um case de Dados e IA Generativa voltado a classificacao textual. O projeto combina:

- pipeline supervisionado baseline para classificacao hierarquica
- camada complementar de IA generativa para refinamento explicavel
- experiencia web orientada a produto
- decisao operacional assistida com base em confianca

O objetivo nao e apenas classificar texto, mas transformar o fluxo em uma solucao demonstravel de ponta a ponta, com separacao clara entre ingestao, analise, modelagem, refinamento e operacao.

## Problema de Negocio

Em cenarios com grande volume de demandas textuais, como atendimento, suporte, financeiro ou cadastro, a triagem manual tende a ser lenta, inconsistente e pouco escalavel. O desafio do case e estruturar uma solucao que:

- receba dados textuais de forma controlada
- compreenda o corpus com EDA
- realize classificacao supervisionada confiavel
- use IA generativa com criterio, e nao como substituta cega do baseline
- apoie a operacao com roteamento assistido e explicabilidade

## Proposta da Solucao

LexiFlow implementa um fluxo em camadas:

1. ingestao do CSV com validacao estrutural
2. analise exploratoria do dataset
3. pre-processamento NLP reutilizavel
4. baseline supervisionado hierarquico
5. refinamento de `classe_detalhada` com GenAI
6. inferencia ponta a ponta em interface web
7. operacao assistida baseada em confianca

O projeto foi pensado para demonstrar maturidade tecnica e narrativa de produto ao mesmo tempo.

## Arquitetura em Camadas

### 1. Ingestao de Dados

- upload de CSV
- validacao de extensao e colunas obrigatorias
- persistencia em `data/raw/`
- suporte a dataset demo para apresentacoes

### 2. Analise Exploratoria

- contagem de registros
- distribuicao de `classe_macro`, `classe_detalhada` e `canal_origem`
- metricas textuais basicas
- visualizacoes geradas em memoria

### 3. Pre-processamento NLP

- normalizacao textual
- limpeza de espacos
- lowercase
- remocao opcional de pontuacao
- remocao opcional de stopwords
- lematizacao opcional
- fallback seguro se o modelo spaCy nao estiver disponivel

### 4. Classificacao Baseline Hierarquica

- `TF-IDF + Logistic Regression`
- nivel 1: previsao de `classe_macro`
- nivel 2: previsao de `classe_detalhada` condicionada pela macro prevista
- persistencia de artefatos com `joblib`

### 5. Refinamento com IA Generativa

- provider configuravel
- suporte a `mock` e `groq`
- prompt estruturado
- fallback seguro para mock
- few-shot contextual com casos similares recuperados do historico

### 6. Operacao Assistida

- leitura da confianca do baseline
- decisao operacional em tres niveis:
  - classificacao automatica
  - classificacao assistida
  - revisao humana
- camada preparada para futuro feedback loop humano

### 7. Evolucoes Futuras

- embeddings reais
- retrieval com banco vetorial
- descoberta de novas classes
- monitoramento de drift
- aprendizagem com feedback humano

## Stack

### Backend

- Flask
- Python
- Jinja Templates

### Dados e NLP

- pandas
- scikit-learn
- spaCy
- numpy

### IA Generativa

- cliente OpenAI-compatible
- Groq
- modo mock para demonstracao sem custo

### Visualizacao e apoio

- matplotlib
- HTML/CSS/JS leve

### Qualidade

- pytest

## Configuracao com `secret.env`

A configuracao principal do projeto fica em `config.py`, com suporte a leitura explicita de `secret.env`.

Ordem de precedencia:

1. variaveis do sistema
2. `secret.env`
3. defaults seguros de desenvolvimento

Exemplo de configuracao:

```env
FLASK_ENV=development
FLASK_APP=run.py
SECRET_KEY=change-me

DEMO_DATASET_PATH=data/demo/lexiflow_demo_dataset.csv
USE_DEMO_DATASET_BY_DEFAULT=false

GENAI_PROVIDER=mock
GENAI_MODEL=mock-model
GENAI_API_KEY=
GENAI_BASE_URL=
GENAI_TEMPERATURE=0.1
GENAI_TIMEOUT_SECONDS=30
GENAI_MOCK_MODE=true

OPENAI_API_KEY=
GROQ_API_KEY=
```

## Uso com Mock

Modo recomendado para desenvolvimento local e apresentacoes sem dependencia de API externa.

```env
GENAI_PROVIDER=mock
GENAI_MOCK_MODE=true
```

Nesse modo:

- nenhuma chave e necessaria
- a interface continua funcional
- a GenAI gera respostas estruturadas para demonstracao

## Uso com Groq

LexiFlow suporta Groq via API OpenAI-compatible.

Exemplo:

```env
GENAI_PROVIDER=groq
GENAI_MODEL=openai/gpt-oss-20b
GENAI_API_KEY=
GROQ_API_KEY=sua-chave-groq
GENAI_BASE_URL=https://api.groq.com/openai/v1
GENAI_TEMPERATURE=0.1
GENAI_TIMEOUT_SECONDS=30
GENAI_MOCK_MODE=false
```

Regras implementadas:

- se `GENAI_PROVIDER=groq` e `GENAI_API_KEY` estiver vazio, a aplicacao tenta `GROQ_API_KEY`
- se a autenticacao falhar, a mensagem de erro e amigavel
- se ocorrer timeout, erro de rede ou erro inesperado, a camada GenAI cai automaticamente para mock

## Como Rodar Localmente

### 1. Ativar o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Opcional: instalar modelo spaCy

```powershell
python -m spacy download pt_core_news_sm
```

### 4. Subir a aplicacao

```powershell
python run.py
```

### 5. Rotas principais

- `/`
- `/arquitetura`
- `/upload`
- `/eda`
- `/baseline`
- `/genai-demo`
- `/predict`
- `/health`

## Estrutura de Pastas

```text
app/
  routes/            Rotas Flask e blueprints
  services/          Regras de negocio, NLP, baseline, GenAI e operacao
  utils/             Utilitarios compartilhados
  templates/         Templates Jinja
  static/            CSS e JS

data/
  raw/               Uploads de entrada
  processed/         Saidas intermediarias
  artifacts/         Modelos e metadados persistidos
  demo/              Dataset demo para apresentacao

tests/               Testes automatizados
config.py            Configuracao central
run.py               Entry point local
secret.env           Variaveis locais do ambiente
```

## Fluxo de Uso

### Caminho 1. Demonstracao com dataset demo

1. configure `USE_DEMO_DATASET_BY_DEFAULT=true` ou selecione `dataset demo` na interface
2. abra `/baseline` e treine os artefatos
3. abra `/predict`
4. cole um novo texto
5. mostre:
   - texto processado
   - macro prevista
   - confianca
   - classe detalhada sugerida
   - casos similares usados como apoio
   - provider usado
   - fluxo operacional recomendado

### Caminho 2. Fluxo real com upload

1. abra `/upload`
2. envie um CSV valido
3. navegue para `/eda`
4. treine em `/baseline`
5. realize inferencia em `/predict`

## Decisoes Tecnicas

### Flask com app factory e blueprints

Permite organizacao por camadas, evolucao modular e estrutura mais profissional que um app monolitico simples.

### TF-IDF + Logistic Regression

Escolha intencional para baseline:

- forte para classificacao textual
- rapido para treinar
- reproduzivel
- explicavel
- adequado para features esparsas

### Hierarquia macro -> detalhada

O problema real foi modelado em dois niveis para refletir a taxonomia de negocio e reduzir o espaco de erro da classe detalhada.

### GenAI complementar, nao substitutiva

A camada generativa nao entra como classificador principal. Ela atua depois do baseline, com contexto controlado, classes permitidas e explicabilidade.

### Few-shot contextual leve

Em vez de infraestrutura pesada, o projeto usa recuperacao local de exemplos similares com `TF-IDF + cosseno`, suficiente para o case e facil de demonstrar.

### Operacao assistida

O resultado da classificacao nao para na predicao. O sistema converte confianca e ambiguidade em uma recomendacao operacional clara.

## Limitacoes

- baseline ainda depende da qualidade do dataset historico
- a camada GenAI nao substitui uma validacao humana em casos ambiguos
- a recuperacao de similares ainda usa TF-IDF, nao embeddings semanticos profundos
- nao ha fila real de feedback humano persistido nesta versao
- monitoramento de drift e descoberta de novas classes ainda sao evolucoes futuras

## Proximos Passos

- persistir feedback humano e override operacional
- adicionar embeddings reais para retrieval semantico
- usar banco vetorial quando o volume justificar
- criar monitoramento de performance e drift
- suportar inferencia em lote
- adicionar dashboard de observabilidade do fluxo

## Por que esta solucao e aderente a vaga de Dados & IA Generativa

Este projeto demonstra, de forma pratica, um conjunto de competencias diretamente aderentes a uma vaga de Dados e IA Generativa:

- modelagem de problema de negocio em pipeline de dados
- NLP aplicado com pre-processamento e classificacao supervisionada
- avaliacao de baseline com metricas e persistencia de artefatos
- integracao de LLM com provider configuravel
- fallback seguro entre ambientes reais e mock
- uso criterioso de GenAI como camada complementar
- few-shot contextual com recuperacao de similares
- visao de produto, interface operacional e experiencia demonstravel

Em outras palavras, LexiFlow nao e apenas um experimento tecnico. Ele mostra capacidade de desenhar uma solucao fim a fim, com equilibrio entre engenharia, IA e aplicacao de negocio.

## Status do README

Este README foi estruturado para:

- leitura limpa no GitHub
- uso em entrevista tecnica
- apoio direto em apresentacao de case
- narrativa clara para perfis de Dados, ML, NLP e IA Generativa
