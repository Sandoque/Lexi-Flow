# LexiFlow

Plataforma de Classificacao Textual Inteligente.

## Visao geral

LexiFlow e uma aplicacao Flask estruturada para um case de classificacao textual com NLP e IA generativa. Esta primeira versao entrega a fundacao do projeto com separacao por camadas, app factory, blueprints, servicos placeholder e interface web inicial.

## Objetivos do produto

1. Receber upload de arquivos CSV.
2. Validar a estrutura dos dados de entrada.
3. Executar analise exploratoria dos textos.
4. Rodar uma classificacao textual baseline.
5. Refinar resultados com IA generativa.
6. Exibir os resultados em interface web.

## Estrutura principal

```text
app/             Aplicacao Flask, rotas, servicos, templates e arquivos estaticos
data/            Dados brutos, processados e artefatos de modelos
notebooks/       Espaco para experimentacao e exploracao analitica
tests/           Testes automatizados do projeto
config.py        Configuracoes por ambiente
run.py           Ponto de entrada local da aplicacao
```

## Como executar

1. Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Instale as dependencias:

```powershell
pip install -r requirements.txt
```

3. Inicie a aplicacao:

```powershell
python run.py
```

## Proximos passos sugeridos

- Implementar persistencia do upload em `data/raw/`.
- Criar regras reais de validacao de schema.
- Adicionar pipeline baseline com vetorizacao e treino.
- Integrar camada de IA generativa com prompts e observabilidade.
- Expandir testes de rotas, servicos e validadores.
