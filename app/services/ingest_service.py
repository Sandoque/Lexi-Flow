"""Servicos para ingestao e validacao inicial de datasets CSV."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.utils.file_handlers import allowed_file, ensure_directory

REQUIRED_COLUMNS = [
    "id_registro",
    "texto",
    "canal_origem",
    "data",
    "classe_macro",
    "classe_detalhada",
]


class IngestaoError(Exception):
    """Representa erros esperados durante a ingestao do dataset."""


def processar_upload_csv(
    uploaded_file: FileStorage,
    upload_folder: str | Path,
    allowed_extensions: set[str] | None = None,
) -> dict:
    """Executa o fluxo completo de ingestao de um arquivo CSV."""
    allowed_extensions = allowed_extensions or {"csv"}
    filename = validar_arquivo_enviado(uploaded_file, allowed_extensions)
    file_bytes = obter_conteudo_arquivo(uploaded_file)
    dataframe = carregar_dataframe(file_bytes)
    validar_colunas_obrigatorias(dataframe)
    saved_path = salvar_arquivo(file_bytes, upload_folder, filename)

    return {
        "saved_path": str(saved_path),
        "summary": gerar_resumo_dataset(saved_path.name, dataframe),
    }


def validar_arquivo_enviado(
    uploaded_file: FileStorage,
    allowed_extensions: set[str],
) -> str:
    """Valida a presenca do arquivo e sua extensao."""
    if uploaded_file is None or uploaded_file.filename is None or uploaded_file.filename == "":
        raise IngestaoError("Selecione um arquivo CSV para continuar.")

    filename = secure_filename(uploaded_file.filename)

    if not filename or not allowed_file(filename):
        raise IngestaoError("O arquivo enviado precisa estar em formato CSV.")

    extension = filename.rsplit(".", 1)[1].lower()
    if extension not in allowed_extensions:
        raise IngestaoError("A extensao do arquivo nao e permitida para ingestao.")

    return filename


def obter_conteudo_arquivo(uploaded_file: FileStorage) -> bytes:
    """Le o conteudo do arquivo enviado para processamento seguro."""
    uploaded_file.stream.seek(0)
    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise IngestaoError("O arquivo enviado esta vazio.")

    return file_bytes


def carregar_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Carrega o CSV em um DataFrame pandas com tratamento de erros."""
    try:
        dataframe = pd.read_csv(BytesIO(file_bytes))
    except (UnicodeDecodeError, ParserError):
        raise IngestaoError("Nao foi possivel ler o arquivo como CSV valido.")
    except EmptyDataError:
        raise IngestaoError("O arquivo CSV nao possui registros para processar.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    if dataframe.empty:
        raise IngestaoError("O arquivo CSV foi lido, mas nao possui linhas de dados.")

    return dataframe


def validar_colunas_obrigatorias(dataframe: pd.DataFrame) -> None:
    """Garante que o dataset possui todas as colunas esperadas pelo case."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise IngestaoError(
            f"O dataset nao possui todas as colunas obrigatorias. Faltam: {missing_text}."
        )


def salvar_arquivo(file_bytes: bytes, upload_folder: str | Path, filename: str) -> Path:
    """Salva o arquivo validado em data/raw sem sobrescrever uploads anteriores."""
    upload_path = ensure_directory(Path(upload_folder))
    destination = gerar_caminho_disponivel(upload_path, filename)
    destination.write_bytes(file_bytes)
    return destination


def gerar_caminho_disponivel(upload_folder: Path, filename: str) -> Path:
    """Gera um nome de arquivo livre caso o nome original ja exista."""
    destination = upload_folder / filename

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1

    while True:
        candidate = upload_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def gerar_resumo_dataset(filename: str, dataframe: pd.DataFrame) -> dict:
    """Monta o resumo exibido na interface apos a ingestao."""
    preview_df = dataframe.head(10).fillna("")
    preview_rows = preview_df.astype(str).to_dict(orient="records")

    return {
        "filename": filename,
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": list(dataframe.columns),
        "preview_columns": list(preview_df.columns),
        "preview_rows": preview_rows,
    }
