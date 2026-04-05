"""Utilitarios para localizar o dataset ativo do projeto."""

from __future__ import annotations

from pathlib import Path


def localizar_dataset_disponivel(
    upload_folder: str | Path,
    preferred_path: str | None = None,
) -> Path:
    """Retorna o arquivo preferencial da sessao ou o CSV mais recente em data/raw."""
    if preferred_path:
        preferred = Path(preferred_path)
        if preferred.exists() and preferred.is_file():
            return preferred

    raw_folder = Path(upload_folder)
    csv_files = sorted(raw_folder.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)

    if not csv_files:
        raise FileNotFoundError("Nenhum CSV disponivel para processamento.")

    return csv_files[0]
