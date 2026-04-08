"""Recuperacao leve de exemplos similares para few-shot contextual no LexiFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.baseline_classifier import BaselineError, carregar_artefatos_baseline
from app.services.genai_refiner import FewShotExample
from app.services.ingest_service import REQUIRED_COLUMNS
from app.services.nlp_config import NLPConfig, obter_configuracao_nlp_padrao
from app.services.preprocessing_service import prepare_texts

DEFAULT_TOP_K = 3


class SimilarExamplesError(Exception):
    """Representa erros esperados na recuperacao de exemplos similares."""


def recuperar_exemplos_similares(
    text: str,
    predicted_macro: str,
    dataset_path: str | Path,
    nlp_config: NLPConfig | None = None,
    top_k: int = DEFAULT_TOP_K,
    restrict_to_macro: bool = True,
) -> dict[str, Any]:
    """Recupera top-k exemplos similares usando TF-IDF e cosseno sobre o historico local."""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return build_empty_similar_examples_result(
            strategy="tfidf_cosine",
            scope="macro" if restrict_to_macro else "global",
        )

    dataframe = carregar_dataset_historico(dataset_path)
    config = nlp_config or obter_configuracao_nlp_padrao()
    processed_df = prepare_texts(dataframe, config)

    filtered_df = filtrar_escopo_historico(
        processed_df=processed_df,
        predicted_macro=predicted_macro,
        restrict_to_macro=restrict_to_macro,
    )
    scope = "macro" if restrict_to_macro and not filtered_df.empty else "global"

    if filtered_df.empty:
        filtered_df = filtrar_escopo_historico(
            processed_df=processed_df,
            predicted_macro=predicted_macro,
            restrict_to_macro=False,
        )

    if filtered_df.empty:
        return build_empty_similar_examples_result(strategy="tfidf_cosine", scope=scope)

    query_df = prepare_texts(pd.DataFrame([{"texto": normalized_text}]), config)
    query_text = str(query_df.loc[0, config.processed_column]).strip()
    if not query_text:
        return build_empty_similar_examples_result(strategy="tfidf_cosine", scope=scope)

    working_df = filtered_df.copy()
    working_df["texto_original"] = working_df[config.original_column].fillna("").astype(str)
    working_df["texto_processado"] = working_df[config.processed_column].fillna("").astype(str)
    working_df["classe_macro"] = working_df["classe_macro"].fillna("").astype(str).str.strip()
    working_df["classe_detalhada"] = working_df["classe_detalhada"].fillna("").astype(str).str.strip()
    working_df = working_df[
        (working_df["texto_processado"] != "")
        & (working_df["classe_macro"] != "")
        & (working_df["classe_detalhada"] != "")
    ].copy()

    if working_df.empty:
        return build_empty_similar_examples_result(strategy="tfidf_cosine", scope=scope)

    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        strip_accents="unicode",
        sublinear_tf=True,
    )
    history_texts = list(working_df["texto_processado"])
    history_matrix = vectorizer.fit_transform(history_texts)
    query_matrix = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_matrix, history_matrix).flatten()

    ranked_indices = sorted(
        range(len(similarities)),
        key=lambda index: float(similarities[index]),
        reverse=True,
    )
    support_examples: list[dict[str, Any]] = []
    few_shot_examples: list[FewShotExample] = []
    selected_signatures: set[tuple[str, str, str]] = set()
    max_examples = max(top_k, 0)

    for index in ranked_indices:
        if len(few_shot_examples) >= max_examples:
            break

        row = working_df.iloc[index]
        signature = construir_assinatura_exemplo(row)
        if signature in selected_signatures:
            continue

        selected_signatures.add(signature)
        similarity = round(float(similarities[index]), 4)
        support_examples.append(
            {
                "text": str(row["texto_original"]),
                "macro_class": str(row["classe_macro"]),
                "detailed_class": str(row["classe_detalhada"]),
                "similarity": similarity,
                "similarity_percent": f"{similarity * 100:.1f}%",
            }
        )
        few_shot_examples.append(
            FewShotExample(
                text=str(row["texto_original"]),
                macro_class=str(row["classe_macro"]),
                detailed_class=str(row["classe_detalhada"]),
                justification=(
                    f"Caso historico com similaridade TF-IDF de {similarity * 100:.1f}% "
                    "em relacao ao texto atual."
                ),
                priority=inferir_prioridade_historica(str(row["texto_original"])),
                ambiguous=False,
            )
        )

    return {
        "few_shot_examples": few_shot_examples,
        "support_examples": support_examples,
        "strategy": "tfidf_cosine",
        "scope": scope,
        "used_count": len(few_shot_examples),
    }


def construir_assinatura_exemplo(row: pd.Series) -> tuple[str, str, str]:
    """Cria uma assinatura estável para evitar duplicatas no few-shot contextual."""
    return (
        str(row["texto_original"]).strip(),
        str(row["classe_macro"]).strip(),
        str(row["classe_detalhada"]).strip(),
    )


def recuperar_exemplos_similares_dos_artefatos(
    text: str,
    predicted_macro: str,
    artifacts_folder: str | Path,
    top_k: int = DEFAULT_TOP_K,
    restrict_to_macro: bool = True,
) -> dict[str, Any]:
    """Conveniencia para buscar exemplos similares a partir dos artefatos do baseline."""
    try:
        artifacts = carregar_artefatos_baseline(artifacts_folder)
    except BaselineError as exc:
        raise SimilarExamplesError("Os artefatos do baseline nao estao disponiveis para recuperar casos similares.") from exc

    metadata = artifacts["metadata"]
    dataset_path = metadata.get("dataset_path")
    if not dataset_path:
        raise SimilarExamplesError("Os artefatos nao informam o dataset historico usado no treino.")

    nlp_config = obter_configuracao_historica(metadata)
    return recuperar_exemplos_similares(
        text=text,
        predicted_macro=predicted_macro,
        dataset_path=dataset_path,
        nlp_config=nlp_config,
        top_k=top_k,
        restrict_to_macro=restrict_to_macro,
    )


def carregar_dataset_historico(dataset_path: str | Path) -> pd.DataFrame:
    """Carrega o dataset historico com validacao minima para recuperar casos similares."""
    try:
        dataframe = pd.read_csv(dataset_path)
    except (UnicodeDecodeError, ParserError):
        raise SimilarExamplesError("O dataset historico nao pode ser interpretado como CSV valido.")
    except EmptyDataError:
        raise SimilarExamplesError("O dataset historico nao possui registros para recuperar casos similares.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise SimilarExamplesError(
            f"O dataset historico nao possui as colunas obrigatorias para few-shot contextual. Faltam: {', '.join(missing_columns)}."
        )

    return dataframe


def filtrar_escopo_historico(
    processed_df: pd.DataFrame,
    predicted_macro: str,
    restrict_to_macro: bool,
) -> pd.DataFrame:
    """Restringe o universo historico a macro prevista quando fizer sentido."""
    if not restrict_to_macro:
        return processed_df.copy()

    normalized_macro = str(predicted_macro).strip()
    return processed_df[
        processed_df["classe_macro"].fillna("").astype(str).str.strip() == normalized_macro
    ].copy()


def obter_configuracao_historica(metadata: dict[str, Any]) -> NLPConfig:
    """Reconstrui a configuracao NLP usada no treino para manter consistencia."""
    raw_config = metadata.get("preprocessing_config")
    if not isinstance(raw_config, dict):
        return obter_configuracao_nlp_padrao()

    allowed_keys = set(NLPConfig.__dataclass_fields__.keys())
    normalized = {key: value for key, value in raw_config.items() if key in allowed_keys}
    return NLPConfig(**normalized)


def inferir_prioridade_historica(text: str) -> str:
    """Infere prioridade leve para enriquecer o contexto few-shot."""
    normalized = text.lower()
    if any(term in normalized for term in ["urgente", "critico", "bloqueado", "imediato"]):
        return "alta"
    if any(term in normalized for term in ["hoje", "prazo", "vencido", "atraso"]):
        return "media"
    return "baixa"


def build_empty_similar_examples_result(strategy: str, scope: str) -> dict[str, Any]:
    """Retorna estrutura vazia padronizada quando nao ha casos similares utilisaveis."""
    return {
        "few_shot_examples": [],
        "support_examples": [],
        "strategy": strategy,
        "scope": scope,
        "used_count": 0,
    }
