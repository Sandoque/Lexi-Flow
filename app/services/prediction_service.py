"""Servico de inferencia ponta a ponta para a experiencia real do LexiFlow."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, cast

import pandas as pd
from sklearn.pipeline import Pipeline

from app.services.baseline_classifier import (
    BaselineError,
    carregar_artefatos_baseline,
    predict_detailed,
)
from app.services.genai_refiner import (
    GenAIRefiner,
    GenAIRefinerError,
    GenAISettings,
    get_demo_few_shot_examples,
)
from app.services.nlp_config import NLPConfig, obter_configuracao_nlp_padrao
from app.services.preprocessing_service import prepare_texts
from app.services.routing_service import classificar_nivel_confianca, definir_fluxo_operacional

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_OPTIONS = [
    "",
    "Email",
    "Chat",
    "Portal",
    "Telefone",
    "WhatsApp",
    "Backoffice",
]


class PredictionError(Exception):
    """Representa erros esperados durante a inferencia ponta a ponta."""


def obter_canais_origem_padrao() -> list[str]:
    """Retorna opcoes simples para o seletor opcional de canal."""
    return DEFAULT_CHANNEL_OPTIONS.copy()


def artefatos_predicao_disponiveis(artifacts_folder: str) -> bool:
    """Indica se os artefatos hierarquicos ja existem para inferencia."""
    try:
        carregar_artefatos_baseline(artifacts_folder)
    except BaselineError:
        return False
    return True


def executar_fluxo_predicao(
    text: str,
    artifacts_folder: str,
    genai_settings: GenAISettings,
    channel_origin: str | None = None,
) -> dict[str, Any]:
    """Executa o fluxo completo de inferencia com baseline e GenAI complementar."""
    normalized_text = str(text or "").strip()
    normalized_channel = str(channel_origin or "").strip()

    if not normalized_text:
        raise PredictionError("Informe um texto para executar a inferencia.")

    artifacts = carregar_artefatos_para_predicao(artifacts_folder)
    nlp_config = obter_configuracao_nlp_da_modelagem(artifacts["metadata"])
    preprocessing_result = aplicar_preprocessamento_predicao(normalized_text, nlp_config)
    processed_text = preprocessing_result["texto_processado"]

    macro_prediction = prever_macro_com_confianca(
        macro_pipeline=cast(Pipeline, artifacts["macro_pipeline"]),
        processed_text=processed_text,
    )
    valid_detailed_classes = obter_classes_detalhadas_validas(
        detailed_artifact=artifacts["detailed_models"],
        macro_prediction=macro_prediction["label"],
    )
    baseline_detail = prever_classe_detalhada_baseline(
        detailed_artifact=artifacts["detailed_models"],
        processed_text=processed_text,
        macro_prediction=macro_prediction["label"],
    )
    genai_result = executar_refinamento_genai(
        text=normalized_text,
        channel_origin=normalized_channel,
        macro_prediction=macro_prediction["label"],
        valid_detailed_classes=valid_detailed_classes,
        baseline_detail=baseline_detail,
        genai_settings=genai_settings,
    )
    operational_routing = definir_fluxo_operacional(
        macro_confidence=macro_prediction["confidence"],
        priority=genai_result["priority"],
        ambiguous_case=genai_result["ambiguous_case"],
        provider=genai_result["provider"],
        genai_status=genai_result["status"],
    )

    return {
        "input": {
            "original_text": normalized_text,
            "processed_text": processed_text,
            "channel_origin": normalized_channel,
        },
        "baseline": {
            "macro_class": macro_prediction["label"],
            "macro_confidence": macro_prediction["confidence"],
            "macro_confidence_percent": macro_prediction["confidence_percent"],
            "confidence_band": macro_prediction["confidence_band"],
            "valid_detailed_classes": valid_detailed_classes,
            "initial_detailed_class": baseline_detail,
            "preprocessing_steps": listar_etapas_preprocessamento(nlp_config),
        },
        "genai": genai_result,
        "recommendation": {
            "action": operational_routing["action"],
            "rationale": operational_routing["reason"],
            "requires_review": operational_routing["review_required"],
            "confidence_band": operational_routing["confidence_level"],
        },
        "routing": operational_routing,
        "artifacts": {
            "macro_pipeline_ready": True,
            "detailed_models_ready": True,
            "trained_at": artifacts["metadata"].get("trained_at"),
            "dataset_path": artifacts["metadata"].get("dataset_path"),
            "dataset_source": artifacts["metadata"].get("dataset_source", "upload"),
            "dataset_source_label": artifacts["metadata"].get("dataset_source_label", "ultimo upload"),
            "dataset_is_demo": bool(artifacts["metadata"].get("dataset_is_demo", False)),
        },
    }


def carregar_artefatos_para_predicao(artifacts_folder: str) -> dict[str, Any]:
    """Carrega artefatos ou orienta o usuario a treinar o baseline primeiro."""
    try:
        return carregar_artefatos_baseline(artifacts_folder)
    except BaselineError as exc:
        raise PredictionError(
            "Os artefatos de inferencia ainda nao existem. Acesse /baseline para treinar o modelo hierarquico."
        ) from exc


def obter_configuracao_nlp_da_modelagem(metadata: dict[str, Any]) -> NLPConfig:
    """Recupera a configuracao NLP usada no treino para manter consistencia na inferencia."""
    raw_config = metadata.get("preprocessing_config")
    if not isinstance(raw_config, dict):
        return obter_configuracao_nlp_padrao()

    allowed_keys = set(NLPConfig.__dataclass_fields__.keys())
    normalized = {key: value for key, value in raw_config.items() if key in allowed_keys}
    return NLPConfig(**normalized)


def aplicar_preprocessamento_predicao(text: str, nlp_config: NLPConfig) -> dict[str, str]:
    """Aplica o mesmo pre-processamento do treino ao texto recebido."""
    dataframe = pd.DataFrame([{"texto": text}])
    processed_df = prepare_texts(dataframe, nlp_config)

    return {
        "texto_original": str(processed_df.loc[0, nlp_config.original_column]),
        "texto_processado": str(processed_df.loc[0, nlp_config.processed_column]),
    }


def prever_macro_com_confianca(macro_pipeline: Pipeline, processed_text: str) -> dict[str, Any]:
    """Gera macroclasse e score de confianca a partir do pipeline hierarquico."""
    labels = [str(prediction) for prediction in macro_pipeline.predict([processed_text])]
    macro_label = labels[0]
    confidence = obter_confianca_macro(macro_pipeline, processed_text, macro_label)

    return {
        "label": macro_label,
        "confidence": confidence,
        "confidence_percent": f"{confidence * 100:.1f}%",
        "confidence_band": classificar_nivel_confianca(confidence),
    }


def obter_confianca_macro(macro_pipeline: Pipeline, processed_text: str, predicted_label: str) -> float:
    """Extrai a confianca da macroclasse quando o estimador suporta probabilidades."""
    if not hasattr(macro_pipeline, "predict_proba"):
        return 0.0

    probability_matrix = macro_pipeline.predict_proba([processed_text])
    if len(probability_matrix) == 0:
        return 0.0

    probabilities = probability_matrix[0]
    classes = [str(label) for label in cast(Any, macro_pipeline.classes_)]
    probability_by_label = {label: float(score) for label, score in zip(classes, probabilities)}
    return round(probability_by_label.get(predicted_label, 0.0), 4)


def obter_classes_detalhadas_validas(detailed_artifact: dict[str, Any], macro_prediction: str) -> list[str]:
    """Recupera as classes detalhadas permitidas para a macro prevista."""
    metadata = cast(list[dict[str, Any]], detailed_artifact.get("metadata", []))

    for item in metadata:
        if str(item.get("macro_class")) == macro_prediction:
            return [str(option) for option in item.get("detail_options", [])]

    models = cast(dict[str, dict[str, Any]], detailed_artifact["models"])
    model_entry = models.get(macro_prediction)
    if model_entry and model_entry.get("type") == "constant":
        return [str(model_entry.get("label", ""))]

    raise PredictionError(
        "Nao foi possivel localizar as classes detalhadas da macro prevista nos artefatos do baseline."
    )


def prever_classe_detalhada_baseline(
    detailed_artifact: dict[str, Any],
    processed_text: str,
    macro_prediction: str,
) -> str:
    """Executa a previsao detalhada inicial antes do refinamento generativo."""
    predictions = predict_detailed(
        detailed_models=cast(dict[str, dict[str, Any]], detailed_artifact["models"]),
        texts=[processed_text],
        macro_predictions=[macro_prediction],
        global_model=cast(dict[str, Any], detailed_artifact["global_model"]),
        fallback_label=str(detailed_artifact["global_fallback_label"]),
        use_macro_filter=True,
    )
    return str(predictions[0])


def executar_refinamento_genai(
    text: str,
    channel_origin: str,
    macro_prediction: str,
    valid_detailed_classes: list[str],
    baseline_detail: str,
    genai_settings: GenAISettings,
) -> dict[str, Any]:
    """Aciona a camada GenAI sem impedir o uso do baseline quando houver degradacao."""
    refiner = GenAIRefiner(genai_settings)
    prompt_text = montar_texto_para_refinamento(text=text, channel_origin=channel_origin)

    try:
        response = refiner.refine(
            text=prompt_text,
            predicted_macro=macro_prediction,
            valid_detailed_classes=valid_detailed_classes,
            few_shot_examples=get_demo_few_shot_examples(),
        )
        result = response["result"]
        return {
            "provider": str(response["provider"]),
            "requested_provider": str(response["requested_provider"]),
            "mode": str(response["mode"]),
            "fallback_reason": response["fallback_reason"],
            "api_key_source": response["api_key_source"],
            "prompt": response["prompt"],
            "baseline_detail": baseline_detail,
            "detailed_class": str(result["detailed_class"]),
            "justification": str(result["justification"]),
            "priority": str(result["priority"]),
            "ambiguous_case": bool(result["ambiguous_case"]),
            "status": "ok",
        }
    except GenAIRefinerError as exc:
        logger.warning("GenAI indisponivel durante a inferencia real: %s", exc)
        return {
            "provider": "indisponivel",
            "requested_provider": genai_settings.requested_provider,
            "mode": "degradado",
            "fallback_reason": str(exc),
            "api_key_source": genai_settings.api_key_source,
            "prompt": None,
            "baseline_detail": baseline_detail,
            "detailed_class": baseline_detail,
            "justification": (
                "A camada GenAI nao respondeu de forma utilizavel. A sugestao detalhada do baseline foi mantida "
                "como apoio operacional."
            ),
            "priority": "nao definida",
            "ambiguous_case": False,
            "status": "baseline_only",
        }


def montar_texto_para_refinamento(text: str, channel_origin: str) -> str:
    """Adiciona contexto operacional leve para a camada de refinamento."""
    if not channel_origin:
        return text
    return f"Canal de origem: {channel_origin}\nTexto do caso: {text}"


def listar_etapas_preprocessamento(config: NLPConfig) -> list[str]:
    """Resume as etapas do NLP usadas na inferencia para auditoria leve."""
    steps: list[str] = []
    config_dict = asdict(config)

    if bool(config_dict["normalize_text"]):
        steps.append("normalizacao")
    if bool(config_dict["clean_whitespace"]):
        steps.append("limpeza de espacos")
    if bool(config_dict["lowercase"]):
        steps.append("lowercase")
    if bool(config_dict["remove_punctuation"]):
        steps.append("remocao de pontuacao")
    if bool(config_dict["remove_stopwords"]):
        steps.append("remocao de stopwords")
    if bool(config_dict["lemmatize"]):
        steps.append("lematizacao")

    steps.append("tokenizacao")
    return steps
