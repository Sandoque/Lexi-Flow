"""Paginas principais da interface do LexiFlow."""

from flask import Blueprint, current_app, jsonify, render_template, request, session

from app.services.ingest_service import REQUIRED_COLUMNS
from app.utils.dataset_locator import resolve_dataset_source

main_bp = Blueprint("main", __name__)


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
            "description": "Recebe CSV, valida o schema obrigatorio e define a base ativa para o restante do fluxo.",
        },
        {
            "order": "02",
            "title": "Analise exploratoria",
            "description": "Resume distribuicoes, canais, classes e sinais iniciais para leitura rapida do dataset.",
        },
        {
            "order": "03",
            "title": "Pre-processamento NLP",
            "description": "Normaliza e estabiliza o texto para manter consistencia entre treino, few-shot e inferencia.",
        },
        {
            "order": "04",
            "title": "Classificacao baseline hierarquica",
            "description": "Treina macroclasse e detalhamento supervisionado com pipeline estatistico reproduzivel.",
        },
        {
            "order": "05",
            "title": "Refinamento com IA generativa",
            "description": "Complementa o baseline com justificativa, prioridade e leitura contextual por provider configuravel.",
        },
        {
            "order": "06",
            "title": "Operacao assistida",
            "description": "Transforma confianca e ambiguidade em recomendacao operacional para automacao ou revisao.",
        },
        {
            "order": "07",
            "title": "Evolucoes futuras",
            "description": "Mantem o case pronto para embeddings, feedback humano, descoberta de classes e monitoramento.",
        },
    ]

    return render_template(
        "index.html",
        solution_layers=solution_layers,
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
            "description": "Recebe CSV, valida schema obrigatorio e prepara o dataset para entrada controlada no fluxo.",
        },
        {
            "order": "02",
            "title": "Analise exploratoria",
            "description": "Resume distribuicoes, classes, canais e sinais textuais para entendimento rapido do corpus.",
        },
        {
            "order": "03",
            "title": "Pre-processamento NLP",
            "description": "Normaliza, tokeniza e estabiliza o texto para garantir consistencia entre treino e inferencia.",
        },
        {
            "order": "04",
            "title": "Classificacao baseline hierarquica",
            "description": "Usa TF-IDF com Logistic Regression para prever macroclasse e restringir o espaco detalhado.",
        },
        {
            "order": "05",
            "title": "Refinamento com IA generativa",
            "description": "Aplica GenAI de forma complementar para explicar e refinar a classe detalhada com critério.",
        },
        {
            "order": "06",
            "title": "Operacao assistida",
            "description": "Converte score, ambiguidade e prioridade em recomendacoes operacionais orientadas a produto.",
        },
        {
            "order": "07",
            "title": "Evolucoes futuras",
            "description": "Prepara a plataforma para incorporar aprendizagem operacional, monitoramento e descoberta de classes.",
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
