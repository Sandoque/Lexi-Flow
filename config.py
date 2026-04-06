"""Modulo central de configuracao da aplicacao LexiFlow."""

import os
import warnings
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SECRET_ENV_PATH = BASE_DIR / "secret.env"
SECRET_ENV_VALUES = dotenv_values(SECRET_ENV_PATH) if SECRET_ENV_PATH.exists() else {}


def bootstrap_environment() -> None:
    """Carrega `secret.env` sem sobrescrever variaveis ja definidas no sistema.

    Ordem de precedencia adotada no projeto:
    1. Variaveis do sistema operacional
    2. Variaveis definidas em `secret.env`
    3. Defaults seguros de desenvolvimento
    """
    if SECRET_ENV_PATH.exists():
        load_dotenv(dotenv_path=SECRET_ENV_PATH, override=False)


def resolve_setting(key: str, default: str | None = None) -> str | None:
    """Resolve uma configuracao respeitando a precedencia do projeto."""
    system_value = os.environ.get(key)
    if system_value not in {None, ""}:
        return system_value

    secret_value = SECRET_ENV_VALUES.get(key)
    if secret_value not in {None, ""}:
        return str(secret_value)

    return default


def resolve_bool_setting(key: str, default: bool) -> bool:
    """Converte uma configuracao booleana com fallback seguro."""
    raw_value = resolve_setting(key)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_int_setting(key: str, default: int) -> int:
    """Converte uma configuracao inteira com fallback seguro."""
    raw_value = resolve_setting(key)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        warnings.warn(f"Valor invalido para {key}. Usando default {default}.", stacklevel=2)
        return default


def resolve_float_setting(key: str, default: float) -> float:
    """Converte uma configuracao numerica com fallback seguro."""
    raw_value = resolve_setting(key)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        warnings.warn(f"Valor invalido para {key}. Usando default {default}.", stacklevel=2)
        return default


def resolve_choice_setting(key: str, default: str, allowed: set[str]) -> str:
    """Valida escolhas conhecidas e cai em default seguro quando necessario."""
    raw_value = resolve_setting(key, default)
    normalized = str(raw_value).strip().lower()

    if normalized in allowed:
        return normalized

    warnings.warn(
        f"Valor invalido para {key}: {raw_value}. Usando default {default}.",
        stacklevel=2,
    )
    return default


bootstrap_environment()


class Config:
    """Configuracoes base compartilhadas entre ambientes."""

    FLASK_ENV = resolve_choice_setting("FLASK_ENV", "development", {"development", "testing", "production"})
    FLASK_APP = resolve_setting("FLASK_APP", "run.py")
    SECRET_KEY = resolve_setting("SECRET_KEY", "dev-secret-key")
    APP_NAME = "LexiFlow"
    APP_SUBTITLE = "Plataforma de Classificacao Textual Inteligente"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "data" / "raw"
    PROCESSED_FOLDER = BASE_DIR / "data" / "processed"
    ARTIFACTS_FOLDER = BASE_DIR / "data" / "artifacts"
    DEMO_DATASET_PATH = Path(resolve_setting("DEMO_DATASET_PATH", str(BASE_DIR / "data" / "demo" / "lexiflow_demo_dataset.csv")))
    USE_DEMO_DATASET_BY_DEFAULT = resolve_bool_setting("USE_DEMO_DATASET_BY_DEFAULT", False)
    ALLOWED_EXTENSIONS = {"csv"}
    OPENAI_API_KEY = resolve_setting("OPENAI_API_KEY")
    GROQ_API_KEY = resolve_setting("GROQ_API_KEY")
    GENAI_PROVIDER = resolve_choice_setting(
        "GENAI_PROVIDER",
        "mock",
        {"mock", "openai", "groq", "openai_compatible"},
    )
    GENAI_MODEL = resolve_setting("GENAI_MODEL", "mock-model")
    GENAI_API_KEY = resolve_setting(
        "GENAI_API_KEY",
        GROQ_API_KEY if GENAI_PROVIDER == "groq" else OPENAI_API_KEY,
    )
    GENAI_BASE_URL = resolve_setting(
        "GENAI_BASE_URL",
        "https://api.groq.com/openai/v1" if GENAI_PROVIDER == "groq" else None,
    )
    GENAI_TEMPERATURE = resolve_float_setting("GENAI_TEMPERATURE", 0.1)
    GENAI_TIMEOUT_SECONDS = resolve_int_setting("GENAI_TIMEOUT_SECONDS", 30)
    GENAI_MOCK_MODE = resolve_bool_setting("GENAI_MOCK_MODE", True)


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
