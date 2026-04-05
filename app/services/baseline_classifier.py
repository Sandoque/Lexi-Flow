"""Servicos para treino hierarquico, avaliacao e persistencia do baseline."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.services.ingest_service import REQUIRED_COLUMNS
from app.services.nlp_config import NLPConfig, obter_configuracao_nlp_padrao
from app.services.preprocessing_service import prepare_texts
from app.utils.chart_builders import gerar_matriz_confusao_base64
from app.utils.dataset_locator import localizar_dataset_disponivel
from app.utils.file_handlers import ensure_directory
from app.utils.text_statistics import resumir_texto

logger = logging.getLogger(__name__)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
MACRO_TARGET = "classe_macro"
DETAILED_TARGET = "classe_detalhada"


class BaselineError(Exception):
    """Representa erros esperados durante o treinamento baseline."""


def get_baseline_placeholder() -> dict:
    """Retorna um resumo inicial para a etapa de modelagem baseline."""
    return {
        "stage": "Classificacao textual baseline",
        "status": "Disponivel",
        "details": "Fluxo hierarquico com macroclasse no nivel 1 e classe detalhada no nivel 2.",
    }


def executar_treinamento_baseline(
    upload_folder: str | Path,
    artifacts_folder: str | Path,
    preferred_path: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    nlp_config: NLPConfig | None = None,
) -> dict:
    """Executa o baseline hierarquico completo em dois niveis."""
    dataset_path = localizar_arquivo_baseline(upload_folder, preferred_path)
    dataframe = carregar_dataset_baseline(dataset_path)
    modeling_df = preparar_dados_modelagem(dataframe, nlp_config=nlp_config)
    split_data = preparar_treino_teste(
        modeling_df=modeling_df,
        target_column=MACRO_TARGET,
        test_size=test_size,
        random_state=random_state,
    )

    macro_artifact = train_macro_classifier(
        train_df=split_data["train_df"],
        random_state=random_state,
    )
    detailed_artifact = train_detailed_classifier(
        train_df=split_data["train_df"],
        random_state=random_state,
    )

    macro_predictions = predict_macro(
        macro_pipeline=macro_artifact["pipeline"],
        texts=list(split_data["test_df"]["texto_processado"].astype(str)),
    )
    detailed_predictions = predict_detailed(
        detailed_models=detailed_artifact["models"],
        texts=list(split_data["test_df"]["texto_processado"].astype(str)),
        macro_predictions=macro_predictions,
        global_model=detailed_artifact["global_model"],
        fallback_label=detailed_artifact["global_fallback_label"],
        use_macro_filter=True,
    )
    oracle_detailed_predictions = predict_detailed(
        detailed_models=detailed_artifact["models"],
        texts=list(split_data["test_df"]["texto_processado"].astype(str)),
        macro_predictions=list(split_data["test_df"][MACRO_TARGET].astype(str)),
        global_model=detailed_artifact["global_model"],
        fallback_label=detailed_artifact["global_fallback_label"],
        use_macro_filter=True,
    )

    macro_evaluation = avaliar_predicoes(
        test_df=split_data["test_df"],
        true_column=MACRO_TARGET,
        predictions=macro_predictions,
        chart_title="Matriz de confusao - macroclasse",
    )
    detailed_evaluation = avaliar_predicoes(
        test_df=split_data["test_df"],
        true_column=DETAILED_TARGET,
        predictions=detailed_predictions,
        chart_title="Matriz de confusao - classe detalhada",
    )
    detailed_oracle_evaluation = avaliar_predicoes(
        test_df=split_data["test_df"],
        true_column=DETAILED_TARGET,
        predictions=oracle_detailed_predictions,
        chart_title="Matriz de confusao - detalhada com macro real",
    )

    refinement_context = montar_contexto_refinamento(detailed_artifact["metadata"])
    artifact_paths = salvar_artefatos_baseline(
        artifacts_folder=artifacts_folder,
        macro_artifact=macro_artifact,
        detailed_artifact=detailed_artifact,
        metadata={
            "trained_at": datetime.now(UTC).isoformat(),
            "dataset_path": str(dataset_path),
            "test_size": test_size,
            "random_state": random_state,
            "split": {
                "train_size": int(split_data["train_df"].shape[0]),
                "test_size": int(split_data["test_df"].shape[0]),
                "stratified": split_data["stratified"],
                "warnings": split_data["warnings"],
            },
            "macro_metrics": macro_evaluation["metrics"],
            "detailed_metrics": detailed_evaluation["metrics"],
            "detailed_oracle_metrics": detailed_oracle_evaluation["metrics"],
            "preprocessing_config": asdict(nlp_config or obter_configuracao_nlp_padrao()),
            "refinement_context": refinement_context,
        },
    )

    return {
        "dataset": {
            "filename": dataset_path.name,
            "path": str(dataset_path),
            "row_count": int(modeling_df.shape[0]),
        },
        "split": {
            "train_size": int(split_data["train_df"].shape[0]),
            "test_size": int(split_data["test_df"].shape[0]),
            "stratified": split_data["stratified"],
            "warnings": split_data["warnings"],
        },
        "hierarchy": {
            "macro": {
                "level": "Nivel 1",
                "title": "Classificacao macro",
                "target": MACRO_TARGET,
                "metrics": macro_evaluation["metrics"],
                "classification_report": macro_evaluation["classification_report"],
                "classification_rows": macro_evaluation["classification_rows"],
                "confusion_matrix_image": macro_evaluation["confusion_matrix_image"],
                "examples": macro_evaluation["examples"],
            },
            "detailed": {
                "level": "Nivel 2",
                "title": "Refinamento detalhado",
                "target": DETAILED_TARGET,
                "metrics": detailed_evaluation["metrics"],
                "classification_report": detailed_evaluation["classification_report"],
                "classification_rows": detailed_evaluation["classification_rows"],
                "confusion_matrix_image": detailed_evaluation["confusion_matrix_image"],
                "examples": detailed_evaluation["examples"],
                "macro_filtered": True,
                "oracle_metrics": detailed_oracle_evaluation["metrics"],
                "oracle_confusion_matrix_image": detailed_oracle_evaluation["confusion_matrix_image"],
                "filter_options": refinement_context["macro_detail_options"],
                "coverage_by_macro": detailed_artifact["metadata"],
            },
            "generative_ready": {
                "stage": "Proximo encaixe",
                "description": "A camada generativa podera atuar apos a previsao macro, usando o conjunto de classes detalhadas permitidas por macro como contexto de refinamento.",
                "hooks": [
                    "macro_predita",
                    "classes_detalhadas_permitidas",
                    "texto_original",
                ],
            },
        },
        "comparison": {
            "macro_accuracy": macro_evaluation["metrics"]["accuracy"],
            "detailed_accuracy": detailed_evaluation["metrics"]["accuracy"],
            "detailed_accuracy_with_true_macro": detailed_oracle_evaluation["metrics"]["accuracy"],
        },
        "artifacts": artifact_paths,
        "limitations": [
            "Erros na macroclasse impactam diretamente a previsao detalhada no fluxo hierarquico.",
            "TF-IDF com regressao logistica e um baseline forte, mas nao modela contexto profundo ou ambiguidade semantica.",
            "A camada detalhada herda a qualidade da taxonomia e da separacao entre classes por macro.",
        ],
    }


def train_macro_classifier(
    train_df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Treina o classificador do nivel 1 para macroclasse."""
    pipeline = construir_pipeline_baseline(random_state=random_state)
    pipeline.fit(train_df["texto_processado"], train_df[MACRO_TARGET])

    return {
        "pipeline": pipeline,
        "target": MACRO_TARGET,
    }


def train_detailed_classifier(
    train_df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Treina classificadores do nivel 2 separados por macroclasse."""
    models: dict[str, dict[str, Any]] = {}
    metadata: list[dict] = []

    for macro_key, group in train_df.groupby(MACRO_TARGET, sort=True):
        macro_class = str(macro_key)
        subset = group[group[DETAILED_TARGET].astype(str).str.strip() != ""].copy()
        detail_labels = subset[DETAILED_TARGET].astype(str)

        if subset.empty:
            continue

        detail_options = [str(option) for option in sorted(list(detail_labels.unique()))]
        if len(detail_options) == 1:
            models[macro_class] = {
                "type": "constant",
                "label": detail_options[0],
            }
        else:
            pipeline = construir_pipeline_baseline(random_state=random_state)
            pipeline.fit(subset["texto_processado"], detail_labels)
            models[macro_class] = {
                "type": "pipeline",
                "model": pipeline,
            }

        metadata.append(
            {
                "macro_class": macro_class,
                "record_count": int(subset.shape[0]),
                "detail_count": len(detail_options),
                "detail_options": detail_options,
            }
        )

    if not models:
        raise BaselineError("Nao foi possivel treinar a camada detalhada com o dataset atual.")

    global_model = treinar_modelo_global_detalhado(train_df, random_state=random_state)
    global_fallback_label = str(
        train_df[DETAILED_TARGET].fillna("").astype(str).str.strip().value_counts().index[0]
    )

    return {
        "models": models,
        "metadata": metadata,
        "global_model": global_model,
        "global_fallback_label": global_fallback_label,
    }


def predict_macro(macro_pipeline: Pipeline, texts: list[str]) -> list[str]:
    """Gera previsoes de macroclasse para o nivel 1."""
    return [str(prediction) for prediction in macro_pipeline.predict(texts)]


def predict_detailed(
    detailed_models: dict[str, dict[str, Any]],
    texts: list[str],
    macro_predictions: list[str],
    global_model: dict[str, Any],
    fallback_label: str,
    use_macro_filter: bool = True,
) -> list[str]:
    """Gera previsoes detalhadas condicionadas pela macroclasse escolhida."""
    predictions: list[str] = []

    for text, macro_prediction in zip(texts, macro_predictions):
        if not use_macro_filter:
            predictions.append(prever_com_modelo_detalhado(global_model, text, fallback_label))
            continue

        model_entry = detailed_models.get(macro_prediction)
        if model_entry is None:
            predictions.append(prever_com_modelo_detalhado(global_model, text, fallback_label))
        else:
            predictions.append(prever_com_modelo_detalhado(model_entry, text, fallback_label))

    return predictions


def localizar_arquivo_baseline(upload_folder: str | Path, preferred_path: str | None = None) -> Path:
    """Resolve o arquivo que sera usado no treino baseline."""
    try:
        return localizar_dataset_disponivel(upload_folder, preferred_path)
    except FileNotFoundError as exc:
        raise BaselineError("Nenhum CSV disponivel para treinamento. Faca um upload antes do baseline.") from exc


def carregar_dataset_baseline(dataset_path: str | Path) -> pd.DataFrame:
    """Carrega o dataset e valida a estrutura minima para modelagem."""
    try:
        dataframe = pd.read_csv(dataset_path)
    except (UnicodeDecodeError, ParserError):
        raise BaselineError("O dataset selecionado nao pode ser interpretado como CSV valido.")
    except EmptyDataError:
        raise BaselineError("O dataset selecionado nao possui registros para treinamento.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise BaselineError(f"O dataset nao possui todas as colunas obrigatorias. Faltam: {missing_text}.")

    if dataframe.empty:
        raise BaselineError("O dataset selecionado esta vazio.")

    return dataframe


def preparar_dados_modelagem(
    dataframe: pd.DataFrame,
    nlp_config: NLPConfig | None = None,
) -> pd.DataFrame:
    """Aplica o pre-processamento textual e prepara a tabela para treino."""
    config = nlp_config or obter_configuracao_nlp_padrao()
    processed_df = prepare_texts(dataframe, config)

    prepared_df = processed_df.copy()
    prepared_df[MACRO_TARGET] = prepared_df[MACRO_TARGET].fillna("").astype(str).str.strip()
    prepared_df[DETAILED_TARGET] = prepared_df[DETAILED_TARGET].fillna("").astype(str).str.strip()
    prepared_df[config.processed_column] = prepared_df[config.processed_column].fillna("").astype(str).str.strip()
    prepared_df["texto_original"] = prepared_df[config.original_column].fillna("").astype(str)

    prepared_df = prepared_df[
        (prepared_df[MACRO_TARGET] != "")
        & (prepared_df[DETAILED_TARGET] != "")
        & (prepared_df[config.processed_column] != "")
    ].copy()

    if prepared_df.empty:
        raise BaselineError("Nao ha registros validos apos o pre-processamento para treinar o baseline.")

    return prepared_df


def preparar_treino_teste(
    modeling_df: pd.DataFrame,
    target_column: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Separa treino e teste de forma reproduzivel e com estratificacao quando possivel."""
    if modeling_df.shape[0] < 6:
        raise BaselineError("Sao necessarios ao menos 6 registros validos para treinar o baseline hierarquico.")

    class_counts = modeling_df[target_column].value_counts()
    if class_counts.shape[0] < 2:
        raise BaselineError("O nivel macro exige pelo menos duas classes distintas para treinamento.")

    warnings: list[str] = []
    stratify = modeling_df[target_column]
    stratified = True
    class_count = int(class_counts.shape[0])
    test_count = math.ceil(modeling_df.shape[0] * test_size)

    if class_counts.min() < 2:
        stratify = None
        stratified = False
        warnings.append(
            "Nem todas as classes macro possuem ao menos 2 exemplos. O split foi feito sem estratificacao."
        )
    elif test_count < class_count:
        stratify = None
        stratified = False
        warnings.append(
            "O conjunto de teste seria menor que o numero de macroclasses. O split foi ajustado sem estratificacao."
        )

    train_df, test_df = train_test_split(
        modeling_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "train_df": train_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "stratified": stratified,
        "warnings": warnings,
    }


def construir_pipeline_baseline(random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    """Constroi o pipeline TF-IDF + Logistic Regression compartilhado."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def treinar_modelo_global_detalhado(
    train_df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Treina um modelo global de classe detalhada para fallback e comparacoes futuras."""
    labels = train_df[DETAILED_TARGET].astype(str)
    unique_labels = sorted(list(labels.unique()))

    if len(unique_labels) == 1:
        return {
            "type": "constant",
            "label": unique_labels[0],
        }

    pipeline = construir_pipeline_baseline(random_state=random_state)
    pipeline.fit(train_df["texto_processado"], labels)
    return {
        "type": "pipeline",
        "model": pipeline,
    }


def prever_com_modelo_detalhado(
    model_entry: dict[str, Any],
    text: str,
    fallback_label: str,
) -> str:
    """Realiza previsao em um modelo detalhado individual ou constante."""
    if model_entry["type"] == "constant":
        return model_entry["label"]
    if model_entry["type"] == "pipeline":
        return model_entry["model"].predict([text])[0]
    return fallback_label


def avaliar_predicoes(
    test_df: pd.DataFrame,
    true_column: str,
    predictions: list[str],
    chart_title: str,
) -> dict:
    """Calcula metricas, exemplos e matriz de confusao para um conjunto de predicoes."""
    labels = sorted(set(list(test_df[true_column].astype(str))) | set(predictions))
    matrix = confusion_matrix(test_df[true_column], predictions, labels=labels)

    metrics = {
        "accuracy": round(float(accuracy_score(test_df[true_column], predictions)), 4),
        "precision": round(
            float(precision_score(test_df[true_column], predictions, average="weighted", zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(test_df[true_column], predictions, average="weighted", zero_division=0)),
            4,
        ),
        "f1_score": round(
            float(f1_score(test_df[true_column], predictions, average="weighted", zero_division=0)),
            4,
        ),
    }

    report_dict = cast(
        dict[str, Any],
        classification_report(
            test_df[true_column],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    )
    report_text = classification_report(
        test_df[true_column],
        predictions,
        labels=labels,
        zero_division=0,
    )

    classification_rows = []
    for label in labels + ["macro avg", "weighted avg"]:
        row = report_dict.get(label)
        if row:
            classification_rows.append(
                {
                    "label": label,
                    "precision": round(float(row["precision"]), 4),
                    "recall": round(float(row["recall"]), 4),
                    "f1_score": round(float(row["f1-score"]), 4),
                    "support": int(row["support"]),
                }
            )

    evaluation_df = test_df[["id_registro", "texto_original", true_column]].copy()
    evaluation_df["predicted"] = predictions
    evaluation_df["is_correct"] = evaluation_df[true_column] == evaluation_df["predicted"]

    return {
        "metrics": metrics,
        "classification_report": report_text,
        "classification_rows": classification_rows,
        "confusion_matrix_image": gerar_matriz_confusao_base64(
            matrix=matrix,
            labels=labels,
            title=chart_title,
        ),
        "examples": {
            "correct": coletar_exemplos_predicao(evaluation_df, true_column, expected_match=True),
            "incorrect": coletar_exemplos_predicao(evaluation_df, true_column, expected_match=False),
        },
    }


def coletar_exemplos_predicao(
    evaluation_df: pd.DataFrame,
    target_column: str,
    expected_match: bool,
    limit: int = 5,
) -> list[dict]:
    """Seleciona exemplos de predicoes corretas ou incorretas para a interface."""
    filtered = evaluation_df[evaluation_df["is_correct"] == expected_match].head(limit)

    return [
        {
            "id_registro": str(row["id_registro"]),
            "expected": row[target_column],
            "predicted": row["predicted"],
            "texto": resumir_texto(row["texto_original"], limit=220),
        }
        for _, row in filtered.iterrows()
    ]


def montar_contexto_refinamento(detailed_metadata: list[dict]) -> dict:
    """Monta o contexto que podera alimentar uma futura camada generativa."""
    macro_detail_options = [
        {
            "macro_class": item["macro_class"],
            "detail_count": item["detail_count"],
            "detail_options": item["detail_options"],
        }
        for item in detailed_metadata
    ]

    return {
        "macro_detail_options": macro_detail_options,
        "refinement_hook": "generative_detail_refinement",
    }


def salvar_artefatos_baseline(
    artifacts_folder: str | Path,
    macro_artifact: dict,
    detailed_artifact: dict,
    metadata: dict,
) -> dict:
    """Salva artefatos do fluxo hierarquico em data/artifacts."""
    artifact_dir = ensure_directory(Path(artifacts_folder))

    macro_pipeline_path = artifact_dir / "baseline_hierarchical_macro_pipeline.joblib"
    macro_vectorizer_path = artifact_dir / "baseline_hierarchical_macro_vectorizer.joblib"
    macro_model_path = artifact_dir / "baseline_hierarchical_macro_model.joblib"
    detailed_models_path = artifact_dir / "baseline_hierarchical_detailed_models.joblib"
    metadata_path = artifact_dir / "baseline_hierarchical_metadata.joblib"

    joblib.dump(macro_artifact["pipeline"], macro_pipeline_path)
    joblib.dump(macro_artifact["pipeline"].named_steps["tfidf"], macro_vectorizer_path)
    joblib.dump(macro_artifact["pipeline"].named_steps["classifier"], macro_model_path)
    joblib.dump(detailed_artifact, detailed_models_path)
    joblib.dump(metadata, metadata_path)

    return {
        "macro_pipeline": str(macro_pipeline_path),
        "macro_vectorizer": str(macro_vectorizer_path),
        "macro_model": str(macro_model_path),
        "detailed_models": str(detailed_models_path),
        "metadata": str(metadata_path),
    }


def carregar_artefatos_baseline(artifacts_folder: str | Path) -> dict:
    """Carrega os artefatos previamente salvos do baseline hierarquico."""
    artifact_dir = Path(artifacts_folder)
    macro_pipeline_path = artifact_dir / "baseline_hierarchical_macro_pipeline.joblib"
    macro_vectorizer_path = artifact_dir / "baseline_hierarchical_macro_vectorizer.joblib"
    macro_model_path = artifact_dir / "baseline_hierarchical_macro_model.joblib"
    detailed_models_path = artifact_dir / "baseline_hierarchical_detailed_models.joblib"
    metadata_path = artifact_dir / "baseline_hierarchical_metadata.joblib"

    paths = [
        macro_pipeline_path,
        macro_vectorizer_path,
        macro_model_path,
        detailed_models_path,
        metadata_path,
    ]
    if not all(path.exists() for path in paths):
        raise BaselineError("Os artefatos hierarquicos ainda nao foram gerados.")

    return {
        "macro_pipeline": joblib.load(macro_pipeline_path),
        "macro_vectorizer": joblib.load(macro_vectorizer_path),
        "macro_model": joblib.load(macro_model_path),
        "detailed_models": joblib.load(detailed_models_path),
        "metadata": joblib.load(metadata_path),
    }
