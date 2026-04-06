"""Fabrica de aplicacao Flask do projeto LexiFlow."""

from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from app.utils.file_handlers import ensure_directory
SECRET_ENV_PATH = Path(__file__).resolve().parent.parent / "secret.env"
load_dotenv(dotenv_path=SECRET_ENV_PATH, override=False)

from config import config_by_name


def create_app(config_name: str = "default") -> Flask:
    """Cria e configura uma instancia da aplicacao Flask."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(config_by_name[config_name])

    prepare_directories(app)
    register_context_processors(app)
    register_blueprints(app)

    return app


def prepare_directories(app: Flask) -> None:
    """Garante que os diretorios operacionais existam ao iniciar a app."""
    ensure_directory(app.config["UPLOAD_FOLDER"])
    ensure_directory(app.config["PROCESSED_FOLDER"])
    ensure_directory(app.config["ARTIFACTS_FOLDER"])
    ensure_directory(Path(app.config["DEMO_DATASET_PATH"]).parent)


def register_context_processors(app: Flask) -> None:
    """Expõe configuracoes globais para os templates Jinja."""

    @app.context_processor
    def inject_app_metadata():
        configured_provider = str(app.config["GENAI_PROVIDER"]).strip().lower()
        return {
            "app_name": app.config["APP_NAME"],
            "app_subtitle": app.config["APP_SUBTITLE"],
            "configured_genai_provider": configured_provider,
            "configured_genai_provider_label": configured_provider.upper() if configured_provider != "mock" else "MOCK",
            "configured_genai_mock_mode": bool(app.config["GENAI_MOCK_MODE"]),
        }


def register_blueprints(app: Flask) -> None:
    """Registra os blueprints de rotas usados pela aplicacao web."""
    from app.routes.main import main_bp
    from app.routes.pipeline import pipeline_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(pipeline_bp)
