"""Paginas principais da interface do LexiFlow."""

from flask import Blueprint, current_app, jsonify, render_template

main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    """Renderiza a pagina inicial com a visao geral do produto."""
    solution_layers = [
        {
            "order": "01",
            "title": "Ingestao",
            "description": "Recebimento de arquivos CSV, verificacao de formato e preparo dos dados de entrada.",
        },
        {
            "order": "02",
            "title": "Exploracao",
            "description": "Leitura inicial dos textos com metricas descritivas e sinais para entendimento do corpus.",
        },
        {
            "order": "03",
            "title": "Classificacao baseline",
            "description": "Pipeline inicial de vetorizacao, treino supervisionado e metricas de referencia.",
        },
        {
            "order": "04",
            "title": "Refinamento GenAI",
            "description": "Ajuste semantico de saidas com apoio de modelos generativos e regras de negocio.",
        },
        {
            "order": "05",
            "title": "Operacao assistida",
            "description": "Camada para revisao humana, acompanhamento operacional e evolucao continua do fluxo.",
        },
    ]

    return render_template("index.html", solution_layers=solution_layers)


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
