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
