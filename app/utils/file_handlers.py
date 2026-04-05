"""Funcoes auxiliares para validacao de arquivos e diretorios."""

from pathlib import Path

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename: str) -> bool:
    """Verifica se o arquivo enviado usa uma extensao permitida."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_directory(path: Path) -> Path:
    """Cria um diretorio quando ele ainda nao existe."""
    path.mkdir(parents=True, exist_ok=True)
    return path
