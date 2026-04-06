"""Servicos para analise exploratoria do ultimo dataset ingerido."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from app.services.ingest_service import REQUIRED_COLUMNS
from app.utils.chart_builders import gerar_grafico_barras_base64
from app.utils.dataset_locator import DatasetSelection, localizar_dataset_disponivel
from app.utils.text_statistics import calcular_estatisticas_textuais, resumir_texto


class EDAError(Exception):
    """Representa erros esperados durante a analise exploratoria."""


def carregar_eda_do_ultimo_dataset(
    upload_folder: str | Path,
    preferred_path: str | None = None,
    demo_dataset_path: str | Path | None = None,
    dataset_source: str | None = None,
    use_demo_by_default: bool = False,
    example_limit: int = 2,
    top_detailed_limit: int = 8,
) -> dict:
    """Carrega o ultimo dataset disponivel e gera sua analise exploratoria."""
    dataset_selection = localizar_dataset_mais_recente(
        upload_folder=upload_folder,
        preferred_path=preferred_path,
        demo_dataset_path=demo_dataset_path,
        dataset_source=dataset_source,
        use_demo_by_default=use_demo_by_default,
    )
    dataset_path = dataset_selection.path

    try:
        dataframe = pd.read_csv(dataset_path)
    except (UnicodeDecodeError, ParserError):
        raise EDAError("O ultimo arquivo disponivel nao pode ser interpretado como CSV valido.")
    except EmptyDataError:
        raise EDAError("O ultimo arquivo disponivel nao possui dados para analise.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    validar_colunas_para_eda(dataframe)

    text_stats = calcular_estatisticas_textuais(dataframe["texto"])
    macro_distribution = (
        dataframe["classe_macro"].fillna("Nao informado").astype(str).value_counts().sort_values(ascending=False)
    )
    detailed_distribution = (
        dataframe["classe_detalhada"]
        .fillna("Nao informado")
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
        .head(top_detailed_limit)
    )
    channel_distribution = (
        dataframe["canal_origem"].fillna("Nao informado").astype(str).value_counts().sort_values(ascending=False)
    )

    return {
        "dataset": {
            "filename": dataset_path.name,
            "absolute_path": str(dataset_path),
            "source": dataset_selection.source,
            "source_label": dataset_selection.source_label,
            "is_demo": dataset_selection.is_demo,
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "columns": list(dataframe.columns),
        },
        "metrics": {
            "total_records": int(dataframe.shape[0]),
            "macro_class_count": int(dataframe["classe_macro"].nunique(dropna=True)),
            "detailed_class_count": int(dataframe["classe_detalhada"].nunique(dropna=True)),
            "avg_text_chars": text_stats["avg_chars"],
            "avg_text_words": text_stats["avg_words"],
            "shortest_text": text_stats["shortest_text"],
            "longest_text": text_stats["longest_text"],
        },
        "distributions": {
            "macro": series_para_items(macro_distribution),
            "detailed_top": series_para_items(detailed_distribution),
            "channel": series_para_items(channel_distribution),
        },
        "charts": {
            "macro": gerar_grafico_barras_base64(
                macro_distribution,
                title="Distribuicao por classe macro",
                color="#1f4f8f",
            ),
            "detailed": gerar_grafico_barras_base64(
                detailed_distribution,
                title="Top classes detalhadas",
                color="#406ea8",
                horizontal=True,
            ),
            "channel": gerar_grafico_barras_base64(
                channel_distribution,
                title="Distribuicao por canal de origem",
                color="#315c47",
            ),
        },
        "examples_by_macro": coletar_exemplos_por_macroclasse(dataframe, limit=example_limit),
    }


def localizar_dataset_mais_recente(
    upload_folder: str | Path,
    preferred_path: str | None = None,
    demo_dataset_path: str | Path | None = None,
    dataset_source: str | None = None,
    use_demo_by_default: bool = False,
) -> DatasetSelection:
    """Localiza o dataset ativo conforme a fonte escolhida para a analise."""
    try:
        return localizar_dataset_disponivel(
            upload_folder=upload_folder,
            preferred_path=preferred_path,
            demo_dataset_path=demo_dataset_path,
            dataset_source=dataset_source,
            use_demo_by_default=use_demo_by_default,
        )
    except FileNotFoundError:
        raise EDAError(
            "Nenhum dataset disponivel para analise. Faca um upload ou configure o dataset demo antes de abrir a EDA."
        )


def validar_colunas_para_eda(dataframe: pd.DataFrame) -> None:
    """Valida se o dataset possui as colunas minimas exigidas para a EDA."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise EDAError(f"O dataset mais recente esta incompleto para EDA. Faltam: {missing_text}.")

    if dataframe.empty:
        raise EDAError("O dataset mais recente nao possui linhas para analise.")


def series_para_items(series: pd.Series) -> list[dict]:
    """Converte uma Series de contagem para lista serializavel em template."""
    return [{"label": str(label), "value": int(value)} for label, value in series.items()]


def coletar_exemplos_por_macroclasse(dataframe: pd.DataFrame, limit: int = 2) -> list[dict]:
    """Seleciona exemplos de texto por macroclasse para a visualizacao da EDA."""
    examples = []
    normalized = dataframe.copy()
    normalized["texto"] = normalized["texto"].fillna("").astype(str)
    normalized["classe_macro"] = normalized["classe_macro"].fillna("Nao informado").astype(str)
    normalized["id_registro"] = normalized["id_registro"].fillna("").astype(str)

    grouped = normalized.groupby("classe_macro", sort=True)

    for macro_class, group in grouped:
        records = []
        for _, row in group.head(limit).iterrows():
            records.append(
                {
                    "id_registro": row["id_registro"] or "-",
                    "texto": resumir_texto(row["texto"], limit=220),
                }
            )

        examples.append(
            {
                "macro_class": macro_class,
                "examples": records,
            }
        )

    return examples
