"""Modulo central de configuracao da aplicacao LexiFlow."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Configuracoes base compartilhadas entre ambientes."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    APP_NAME = "LexiFlow"
    APP_SUBTITLE = "Plataforma de Classificacao Textual Inteligente"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "data" / "raw"
    PROCESSED_FOLDER = BASE_DIR / "data" / "processed"
    ARTIFACTS_FOLDER = BASE_DIR / "data" / "artifacts"
    ALLOWED_EXTENSIONS = {"csv"}
    GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "mock")
    GENAI_MODEL = os.getenv("GENAI_MODEL", "mock-model")
    GENAI_API_KEY = os.getenv("GENAI_API_KEY")
    GENAI_BASE_URL = os.getenv("GENAI_BASE_URL")
    GENAI_TEMPERATURE = float(os.getenv("GENAI_TEMPERATURE", "0.1"))
    GENAI_TIMEOUT_SECONDS = int(os.getenv("GENAI_TIMEOUT_SECONDS", "30"))
    GENAI_MOCK_MODE = os.getenv("GENAI_MOCK_MODE", "true").lower() in {"1", "true", "yes", "on"}
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class DevelopmentConfig(Config):
    """Configuracao usada durante o desenvolvimento local."""

    DEBUG = True


class TestingConfig(Config):
    """Configuracao usada em testes automatizados."""

    TESTING = True


class ProductionConfig(Config):
    """Configuracao usada em ambientes de producao."""

    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
