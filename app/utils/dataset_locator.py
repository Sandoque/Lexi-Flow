"""Utilitarios para localizar o dataset ativo do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetSelection:
    """Representa a fonte de dados efetivamente usada pela aplicacao."""

    path: Path
    source: str
    source_label: str
    is_demo: bool


def resolve_dataset_source(
    requested_source: str | None = None,
    current_source: str | None = None,
    use_demo_by_default: bool = False,
) -> str:
    """Resolve a fonte ativa entre upload e demo com fallback previsivel."""
    normalized_requested = normalize_dataset_source(requested_source)
    if normalized_requested:
        return normalized_requested

    normalized_current = normalize_dataset_source(current_source)
    if normalized_current:
        return normalized_current

    return "demo" if use_demo_by_default else "upload"


def localizar_dataset_disponivel(
    upload_folder: str | Path,
    preferred_path: str | None = None,
    demo_dataset_path: str | Path | None = None,
    dataset_source: str | None = None,
    use_demo_by_default: bool = False,
) -> DatasetSelection:
    """Retorna o dataset ativo conforme a fonte configurada para a execucao."""
    active_source = resolve_dataset_source(
        requested_source=dataset_source,
        current_source=None,
        use_demo_by_default=use_demo_by_default,
    )

    if active_source == "demo":
        return localizar_dataset_demo(demo_dataset_path)

    return localizar_dataset_upload(upload_folder, preferred_path)


def localizar_dataset_upload(
    upload_folder: str | Path,
    preferred_path: str | None = None,
) -> DatasetSelection:
    """Localiza o ultimo upload valido disponivel em data/raw."""
    if preferred_path:
        preferred = Path(preferred_path)
        if preferred.exists() and preferred.is_file():
            return DatasetSelection(
                path=preferred,
                source="upload",
                source_label="ultimo upload",
                is_demo=False,
            )

    raw_folder = Path(upload_folder)
    csv_files = sorted(raw_folder.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)

    if not csv_files:
        raise FileNotFoundError("Nenhum CSV disponivel para processamento.")

    return DatasetSelection(
        path=csv_files[0],
        source="upload",
        source_label="ultimo upload",
        is_demo=False,
    )


def localizar_dataset_demo(demo_dataset_path: str | Path | None) -> DatasetSelection:
    """Localiza o dataset demo configurado pela aplicacao."""
    if demo_dataset_path is None:
        raise FileNotFoundError("Nenhum dataset demo foi configurado.")

    demo_path = Path(demo_dataset_path)
    if not demo_path.exists() or not demo_path.is_file():
        raise FileNotFoundError("O dataset demo configurado nao foi encontrado.")

    return DatasetSelection(
        path=demo_path,
        source="demo",
        source_label="dataset demo",
        is_demo=True,
    )


def normalize_dataset_source(value: str | None) -> str | None:
    """Normaliza a fonte de dataset aceita pela aplicacao."""
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"upload", "demo"}:
        return normalized

    return None
