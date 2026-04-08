# LexiFlow

Plataforma de Classificação Textual Inteligente.

## Visão Geral

LexiFlow é uma aplicação Flask estruturada como um case de Dados e IA Generativa para a XPTO Data Solutions. O projeto parte de um problema operacional real: a empresa recebe diariamente textos de suporte, solicitações de clientes, relatos operacionais e feedbacks textuais sem categorização automática.

A solução foi desenhada para ir além de um classificador isolado. O fluxo combina:

- ingestão controlada de dados
- análise exploratória do corpus
- pré-processamento NLP reutilizável
- baseline supervisionado hierárquico
- refinamento complementar com IA generativa
- recomendação operacional assistida por confiança

O objetivo não é apenas prever classes, mas transformar a triagem textual em um fluxo demonstrável de ponta a ponta, com narrativa clara de produto, explicabilidade e evolução futura.

## Problema de Negócio

A XPTO Data Solutions atua com soluções de dados, automação e inteligência artificial para empresas de médio porte em setores como logística, serviços financeiros e varejo digital.

No contexto do case, a empresa recebe registros textuais vindos de múltiplos canais, como:

- descrições de chamados de suporte
- solicitações de clientes
- relatos operacionais
- feedbacks textuais sobre serviços

Sem categorização automática, esse fluxo gera dores operacionais relevantes:

- dificuldade de priorização
- aumento do tempo de resposta
- pouca visibilidade por tipo de problema
- dependência excessiva de leitura manual

A proposta do LexiFlow é responder a esse cenário com uma solução técnica viável, explicável e preparada para crescer.

## Proposta da Solução

O LexiFlow organiza o problema em camadas:

1. ingestão do CSV com validação estrutural
2. análise exploratória do dataset
3. pré-processamento textual com NLP
4. classificação baseline hierárquica
5. refinamento da classe detalhada com IA generativa
6. inferência ponta a ponta em interface web
7. operação assistida baseada em confiança

Essa abordagem equilibra baseline supervisionado e GenAI com critério. O baseline continua sendo a referência principal de classificação, enquanto a camada generativa entra como refinamento contextual, explicabilidade e apoio operacional.

## Impacto Esperado

Mais do que prever classes, a solução foi pensada para apoiar a operação da XPTO com resultados claros:

- priorização mais rápida de casos críticos
- redução da triagem manual repetitiva
- visibilidade por tipo de problema e canal de origem
- apoio à decisão para filas, SLA e revisão humana

## Arquitetura em Camadas

### 1. Ingestão de Dados

- upload de CSV
- validação de extensão e colunas obrigatórias
- persistência em `data/raw/`
- suporte a dataset demo para apresentações

### 2. Análise Exploratória

- contagem de registros
- distribuição de `classe_macro`, `classe_detalhada` e `canal_origem`
- métricas textuais básicas
- visualizações simples para leitura executiva e técnica

### 3. Pré-processamento NLP

- normalização textual
- limpeza de espaços
- lowercase
- remoção opcional de pontuação
- remoção opcional de stopwords
- lematização opcional
- fallback seguro se o modelo spaCy não estiver disponível

### 4. Classificação Baseline Hierárquica

- `TF-IDF + Logistic Regression`
- nível 1: previsão de `classe_macro`
- nível 2: previsão de `classe_detalhada` condicionada pela macro prevista
- persistência de artefatos com `joblib`

### 5. Refinamento com IA Generativa

- provider configurável
- suporte a `mock` e `groq`
- prompt estruturado
- fallback seguro para mock
- few-shot contextual com casos similares recuperados do histórico

### 6. Operação Assistida

- leitura da confiança do baseline
- decisão operacional em três níveis:
  - classificação automática
  - classificação assistida
  - revisão humana
- estrutura preparada para futuro feedback humano

### 7. Evoluções Futuras

- embeddings semânticos reais
- recuperação com banco vetorial
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
- modo mock para demonstração sem custo

### Visualização e apoio

- matplotlib
- HTML, CSS e JavaScript leve

### Qualidade

- pytest

## Configuração com `secret.env`

A configuração principal do projeto fica em `config.py`, com suporte explícito a `secret.env`.

Ordem de precedência:

1. variáveis do sistema
2. `secret.env`
3. defaults seguros para desenvolvimento

Exemplo:

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

Modo recomendado para desenvolvimento local e apresentações sem depender de API externa.

```env
GENAI_PROVIDER=mock
GENAI_MOCK_MODE=true
```

Nesse modo:

- nenhuma chave é necessária
- a interface continua funcional
- a camada GenAI retorna respostas estruturadas para demonstração

## Uso com Groq

O LexiFlow suporta Groq por meio de API OpenAI-compatible.

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

- se `GENAI_PROVIDER=groq` e `GENAI_API_KEY` estiver vazio, a aplicação tenta `GROQ_API_KEY`
- se a autenticação falhar, a mensagem é amigável
- se ocorrer timeout, erro de rede ou erro inesperado, a camada GenAI cai automaticamente para mock

## Como Rodar Localmente

### 1. Ativar o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 3. Opcional: instalar modelo spaCy

```powershell
python -m spacy download pt_core_news_sm
```

### 4. Subir a aplicação

```powershell
python run.py
```

### 5. Rotas principais

- `/`
- `/eda`
- `/baseline`
- `/predict`
- `/genai-demo`
- `/arquitetura`

Rotas de apoio:

- `/upload`
- `/results`
- `/health`

## Estrutura de Pastas

```text
app/
  routes/            Rotas Flask e blueprints
  services/          Regras de negócio, NLP, baseline, GenAI e operação
  utils/             Utilitários compartilhados
  templates/         Templates Jinja
  static/            CSS e JS

data/
  raw/               Uploads de entrada
  processed/         Saídas intermediárias
  artifacts/         Modelos e metadados persistidos
  demo/              Dataset demo para apresentação

notebooks/           Apêndice analítico do case
tests/               Testes automatizados
config.py            Configuração central
run.py               Entry point local
secret.env           Variáveis locais do ambiente
```

## Fluxo de Uso

### Caminho 1. Demonstração com dataset demo

1. abra `/` e mantenha o dataset demo como fonte
2. avance para `/eda` para contextualizar o corpus
3. treine o baseline em `/baseline`
4. siga para `/predict`
5. cole um novo texto
6. apresente:
   - macro prevista
   - confiança
   - classe detalhada sugerida
   - provider usado
   - fluxo operacional recomendado

### Caminho 2. Fluxo com upload real

1. abra `/`
2. envie um CSV válido na área de ingestão da home ou pela rota `/upload`
3. navegue para `/eda`
4. treine em `/baseline`
5. execute a inferência em `/predict`

## Decisões Técnicas

### Defesa técnica resumida

Se a solução precisar ser defendida de forma curta em entrevista, a lógica central é:

- baseline supervisionado primeiro para garantir referência forte, explicável e reproduzível
- hierarquia macro -> detalhada para refletir a taxonomia real do problema
- GenAI depois do baseline para refinar, justificar e apoiar a operação sem substituir a classificação principal

### Flask com app factory e blueprints

Permite organização por camadas, evolução modular e estrutura mais profissional do que um app monolítico simples.

### TF-IDF + Logistic Regression

Escolha intencional para baseline porque:

- é forte para classificação textual
- treina rápido
- é reproduzível
- é explicável
- funciona bem com features esparsas

### Hierarquia macro -> detalhada

O problema foi modelado em dois níveis para refletir a taxonomia de negócio e reduzir o espaço de erro da classe detalhada.

### GenAI complementar, não substitutiva

A camada generativa não entra como classificador principal. Ela atua depois do baseline, com contexto controlado, classes válidas e justificativa curta.

### Few-shot contextual leve

Em vez de depender de infraestrutura pesada, o projeto usa recuperação local de exemplos similares com `TF-IDF + cosseno`, suficiente para o case e fácil de demonstrar.

### Operação assistida

O resultado da classificação não para na predição. O sistema converte confiança e ambiguidade em uma recomendação operacional clara.

## Limitações

- erros na macroclasse impactam diretamente o detalhamento
- o baseline depende da qualidade do dataset histórico
- classes com pouco volume tendem a ser mais sensíveis a confusão
- a camada GenAI não substitui validação humana em casos ambíguos
- a recuperação de similares ainda usa TF-IDF, não embeddings profundos
- ainda não há feedback humano persistido nesta versão
- monitoramento de drift e descoberta de novas classes permanecem como evolução futura

## Próximos Passos

- persistir feedback humano e override operacional
- adicionar embeddings reais para retrieval semântico
- incorporar banco vetorial quando o volume justificar
- criar monitoramento de performance e drift
- suportar inferência em lote
- adicionar observabilidade do fluxo

## Apêndice Analítico

O repositório inclui um apêndice leve em `notebooks/` para apoiar a conversa técnica sobre:

- leitura inicial do dataset
- hipóteses de modelagem
- justificativa do baseline
- interpretação dos resultados

## Por Que Esta Solução É Aderente à Vaga de Dados & IA Generativa

Este projeto demonstra, de forma prática, competências diretamente aderentes a uma vaga de Dados e IA Generativa:

- entendimento de problema de negócio com tradução para solução analítica
- EDA aplicada a dados textuais
- NLP com pré-processamento reutilizável
- classificação supervisionada com baseline defendível
- avaliação de métricas e leitura crítica dos erros
- integração pragmática de IA generativa em fluxo real
- preocupação com operação assistida, explicabilidade e evolução

Em outras palavras, o LexiFlow não é apenas um experimento técnico. Ele mostra capacidade de estruturar uma solução fim a fim, equilibrando engenharia, modelagem, IA generativa e aplicação de negócio.
