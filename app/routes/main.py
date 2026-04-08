"""Paginas principais da interface do LexiFlow."""

from flask import Blueprint, current_app, jsonify, render_template, request, session

from app.services.ingest_service import REQUIRED_COLUMNS
from app.utils.dataset_locator import resolve_dataset_source

main_bp = Blueprint("main", __name__)

CASE_CONTEXT_CARDS = [
    {
        "title": "Entradas textuais da XPTO",
        "description": (
            "Chamados de suporte, solicitacoes de clientes, relatos operacionais e feedbacks "
            "textuais chegam por email, portal, chat, telefone e canais internos."
        ),
    },
    {
        "title": "Dores do processo manual",
        "description": (
            "Sem categorizacao automatica, a operacao perde velocidade, prioriza pior os casos "
            "mais criticos e enxerga pouco os principais tipos de problema."
        ),
    },
    {
        "title": "Objetivo do case",
        "description": (
            "Classificar textos com criterio para apoiar triagem, SLA, metricas por tipo de "
            "ocorrencia e evolucao dos produtos de dados da XPTO."
        ),
    },
]

BUSINESS_IMPACTS = [
    {
        "title": "Priorizacao mais rapida",
        "description": "Casos criticos deixam de competir com demandas simples na fila operacional.",
    },
    {
        "title": "Menos triagem manual",
        "description": "A equipe reduz leitura repetitiva e concentra energia nos casos ambiguos.",
    },
    {
        "title": "Mais visibilidade",
        "description": "A empresa passa a enxergar volume, recorrencia e origem dos principais problemas.",
    },
    {
        "title": "Apoio a decisao",
        "description": "O fluxo vira um produto de dados para orientar resposta, priorizacao e melhoria continua.",
    },
]


def obter_fonte_dataset_inicial() -> str:
    """Resolve a fonte de dados para a home e preserva a escolha na sessao."""
    dataset_source = resolve_dataset_source(
        requested_source=request.args.get("dataset_source"),
        current_source=session.get("dataset_source"),
        use_demo_by_default=bool(current_app.config["USE_DEMO_DATASET_BY_DEFAULT"]),
    )
    session["dataset_source"] = dataset_source
    return dataset_source


@main_bp.get("/")
def index():
    """Renderiza a pagina inicial com a visao geral do produto."""
    dataset_source = obter_fonte_dataset_inicial()
    solution_layers = [
        {
            "order": "01",
            "title": "Ingestao de dados",
            "description": (
                "Consolida textos de suporte, clientes, operacao e feedback em uma entrada "
                "controlada para o fluxo analitico."
            ),
        },
        {
            "order": "02",
            "title": "Analise exploratoria",
            "description": (
                "Resume canais, classes e recorrencias para acelerar a leitura inicial do corpus "
                "e orientar a modelagem."
            ),
        },
        {
            "order": "03",
            "title": "Pre-processamento NLP",
            "description": (
                "Normaliza a linguagem heterogenea dos registros para manter consistencia entre "
                "treino, similaridade e inferencia."
            ),
        },
        {
            "order": "04",
            "title": "Classificacao baseline hierarquica",
            "description": (
                "Estrutura a triagem em dois niveis: macroproblema e motivo detalhado, refletindo "
                "a taxonomia operacional."
            ),
        },
        {
            "order": "05",
            "title": "Refinamento com IA generativa",
            "description": (
                "Complementa o baseline com justificativa curta, prioridade e leitura contextual "
                "com provider configuravel."
            ),
        },
        {
            "order": "06",
            "title": "Operacao assistida",
            "description": (
                "Transforma confianca e ambiguidade em recomendacao operacional para automacao, "
                "assistencia ou revisao humana."
            ),
        },
        {
            "order": "07",
            "title": "Evolucoes futuras",
            "description": (
                "Mantem o case pronto para embeddings, feedback humano, descoberta de classes "
                "e monitoramento."
            ),
        },
    ]

    return render_template(
        "index.html",
        solution_layers=solution_layers,
        case_context_cards=CASE_CONTEXT_CARDS,
        business_impacts=BUSINESS_IMPACTS,
        expected_columns=REQUIRED_COLUMNS,
        dataset_source=dataset_source,
        dataset_source_label="dataset demo" if dataset_source == "demo" else "ultimo upload",
        using_demo_dataset=dataset_source == "demo",
        demo_dataset_path=str(current_app.config["DEMO_DATASET_PATH"]),
    )


@main_bp.get("/arquitetura")
def architecture():
    """Renderiza a visao arquitetural do case para apresentacao executiva e tecnica."""
    architecture_layers = [
        {
            "order": "01",
            "title": "Ingestao de dados",
            "description": (
                "Recebe chamados, solicitacoes, relatos e feedbacks em CSV, valida o schema e "
                "define uma entrada controlada para o fluxo."
            ),
        },
        {
            "order": "02",
            "title": "Analise exploratoria",
            "description": (
                "Resume distribuicoes por canal, classe e recorrencia para acelerar o entendimento "
                "do corpus recebido pela XPTO."
            ),
        },
        {
            "order": "03",
            "title": "Pre-processamento NLP",
            "description": (
                "Normaliza, tokeniza e estabiliza textos heterogeneos de clientes e times internos "
                "para garantir consistencia entre treino e inferencia."
            ),
        },
        {
            "order": "04",
            "title": "Classificacao baseline hierarquica",
            "description": (
                "Usa TF-IDF com Logistic Regression para prever macroclasse e restringir o espaco "
                "de motivos detalhados da triagem."
            ),
        },
        {
            "order": "05",
            "title": "Refinamento com IA generativa",
            "description": (
                "Aplica GenAI de forma complementar para explicar, refinar e contextualizar a "
                "classe detalhada com criterio."
            ),
        },
        {
            "order": "06",
            "title": "Operacao assistida",
            "description": (
                "Converte score, ambiguidade e prioridade em recomendacoes operacionais orientadas "
                "a SLA, fila e decisao."
            ),
        },
        {
            "order": "07",
            "title": "Evolucoes futuras",
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
        roadmap_items=roadmap_items,
    )


@main_bp.get("/health")
def health():
    """Retorna o status basico da aplicacao para monitoramento."""
    return jsonify(
        {
            "status": "ok",
            "application": current_app.config["APP_NAME"],
            "environment": "development" if current_app.debug else "production",
        }
    )
