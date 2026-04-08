"""Páginas principais da interface do LexiFlow."""

from flask import Blueprint, current_app, jsonify, render_template, request, session

from app.services.ingest_service import REQUIRED_COLUMNS
from app.utils.dataset_locator import resolve_dataset_source

main_bp = Blueprint("main", __name__)

CASE_CONTEXT_CARDS = [
    {
        "title": "Entradas textuais da XPTO",
        "description": (
            "Chamados de suporte, solicitações de clientes, relatos operacionais e feedbacks "
            "textuais chegam por e-mail, portal, chat, telefone e canais internos."
        ),
    },
    {
        "title": "Dores do processo manual",
        "description": (
            "Sem categorização automática, a operação perde velocidade, prioriza pior os casos "
            "mais críticos e enxerga pouco os principais tipos de problema."
        ),
    },
    {
        "title": "Objetivo do case",
        "description": (
            "Classificar textos com critério para apoiar triagem, SLA, métricas por tipo de "
            "ocorrência e evolução dos produtos de dados da XPTO."
        ),
    },
]

BUSINESS_IMPACTS = [
    {
        "title": "Priorização mais rápida",
        "description": "Casos críticos deixam de competir com demandas simples na fila operacional.",
    },
    {
        "title": "Menos triagem manual",
        "description": "A equipe reduz leitura repetitiva e concentra energia nos casos ambíguos.",
    },
    {
        "title": "Mais visibilidade",
        "description": "A empresa passa a enxergar volume, recorrência e origem dos principais problemas.",
    },
    {
        "title": "Apoio à decisão",
        "description": "O fluxo vira um produto de dados para orientar resposta, priorização e melhoria contínua.",
    },
]

TECHNICAL_DECISIONS = [
    {
        "title": "Por que baseline supervisionado primeiro",
        "description": (
            "Porque o case pede uma estratégia de classificação defensável, com treino rápido, avaliação clara "
            "e capacidade de explicar o comportamento do modelo."
        ),
    },
    {
        "title": "Por que modelar em dois níveis",
        "description": (
            "Porque a triagem real depende de taxonomia: primeiro entender a natureza do problema, depois "
            "escolher o motivo detalhado dentro desse contexto."
        ),
    },
    {
        "title": "Por que GenAI entra depois",
        "description": (
            "Porque a camada generativa agrega explicabilidade, prioridade e refinamento contextual sem substituir "
            "o baseline supervisionado como referência principal."
        ),
    },
]


def obter_fonte_dataset_inicial() -> str:
    """Resolve a fonte de dados para a home e preserva a escolha na sessão."""
    dataset_source = resolve_dataset_source(
        requested_source=request.args.get("dataset_source"),
        current_source=session.get("dataset_source"),
        use_demo_by_default=bool(current_app.config["USE_DEMO_DATASET_BY_DEFAULT"]),
    )
    session["dataset_source"] = dataset_source
    return dataset_source


@main_bp.get("/")
def index():
    """Renderiza a página inicial com a visão geral do produto."""
    dataset_source = obter_fonte_dataset_inicial()
    solution_layers = [
        {
            "order": "01",
            "title": "Ingestão de dados",
            "description": (
                "Consolida textos de suporte, clientes, operação e feedback em uma entrada "
                "controlada para o fluxo analítico."
            ),
        },
        {
            "order": "02",
            "title": "Análise exploratória",
            "description": (
                "Resume canais, classes e recorrências para acelerar a leitura inicial do corpus "
                "e orientar a modelagem."
            ),
        },
        {
            "order": "03",
            "title": "Pre-processamento NLP",
            "description": (
                "Normaliza a linguagem heterogênea dos registros para manter consistência entre "
                "treino, similaridade e inferência."
            ),
        },
        {
            "order": "04",
            "title": "Classificação baseline hierárquica",
            "description": (
                "Estrutura a triagem em dois níveis: macroproblema e motivo detalhado, refletindo "
                "a taxonomia operacional."
            ),
        },
        {
            "order": "05",
            "title": "Refinamento com IA generativa",
            "description": (
                "Complementa o baseline com justificativa curta, prioridade e leitura contextual "
                "com provider configurável."
            ),
        },
        {
            "order": "06",
            "title": "Operação assistida",
            "description": (
                "Transforma confiança e ambiguidade em recomendação operacional para automação, "
                "assistência ou revisão humana."
            ),
        },
        {
            "order": "07",
            "title": "Evoluções futuras",
            "description": (
                "Mantém o case pronto para embeddings, feedback humano, descoberta de classes "
                "e monitoramento."
            ),
        },
    ]

    return render_template(
        "index.html",
        solution_layers=solution_layers,
        case_context_cards=CASE_CONTEXT_CARDS,
        business_impacts=BUSINESS_IMPACTS,
        technical_decisions=TECHNICAL_DECISIONS,
        expected_columns=REQUIRED_COLUMNS,
        dataset_source=dataset_source,
        dataset_source_label="dataset demo" if dataset_source == "demo" else "último upload",
        using_demo_dataset=dataset_source == "demo",
        demo_dataset_path=str(current_app.config["DEMO_DATASET_PATH"]),
    )


@main_bp.get("/arquitetura")
def architecture():
    """Renderiza a visão arquitetural do case para apresentação executiva e técnica."""
    architecture_layers = [
        {
            "order": "01",
            "title": "Ingestão de dados",
            "description": (
                "Recebe chamados, solicitações, relatos e feedbacks em CSV, valida o schema e "
                "define uma entrada controlada para o fluxo."
            ),
        },
        {
            "order": "02",
            "title": "Análise exploratória",
            "description": (
                "Resume distribuições por canal, classe e recorrência para acelerar o entendimento "
                "do corpus recebido pela XPTO."
            ),
        },
        {
            "order": "03",
            "title": "Pre-processamento NLP",
            "description": (
                "Normaliza, tokeniza e estabiliza textos heterogêneos de clientes e times internos "
                "para garantir consistência entre treino e inferência."
            ),
        },
        {
            "order": "04",
            "title": "Classificação baseline hierárquica",
            "description": (
                "Usa TF-IDF com Logistic Regression para prever macroclasse e restringir o espaço "
                "de motivos detalhados da triagem."
            ),
        },
        {
            "order": "05",
            "title": "Refinamento com IA generativa",
            "description": (
                "Aplica GenAI de forma complementar para explicar, refinar e contextualizar a "
                "classe detalhada com critério."
            ),
        },
        {
            "order": "06",
            "title": "Operação assistida",
            "description": (
                "Converte score, ambiguidade e prioridade em recomendações operacionais orientadas "
                "a SLA, fila e decisão."
            ),
        },
        {
            "order": "07",
            "title": "Evoluções futuras",
            "description": (
                "Prepara a plataforma para incorporar aprendizagem operacional, monitoramento de "
                "drift e descoberta de novas classes."
            ),
        },
    ]
    roadmap_items = [
        "Few-shot contextual",
        "Embeddings",
        "Fila humana com feedback loop",
        "Descoberta de novas classes",
        "Monitoramento de drift",
    ]

    return render_template(
        "architecture.html",
        architecture_layers=architecture_layers,
        case_context_cards=CASE_CONTEXT_CARDS,
        business_impacts=BUSINESS_IMPACTS,
        technical_decisions=TECHNICAL_DECISIONS,
        roadmap_items=roadmap_items,
    )


@main_bp.get("/health")
def health():
    """Retorna o status básico da aplicação para monitoramento."""
    return jsonify(
        {
            "status": "ok",
            "application": current_app.config["APP_NAME"],
            "environment": "development" if current_app.debug else "production",
        }
    )
