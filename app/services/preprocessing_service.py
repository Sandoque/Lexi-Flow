"""Pipeline reutilizavel de pre-processamento textual para classificacao."""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from app.services.nlp_config import NLPConfig, obter_configuracao_nlp_padrao

logger = logging.getLogger(__name__)

FALLBACK_STOPWORDS_PT = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "sem",
    "um",
    "uma",
}

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class NLPResources:
    """Recursos carregados para o pipeline, com fallback seguro."""

    nlp: Any | None
    stopwords: set[str]
    backend: str
    lemma_available: bool


def prepare_texts(df: pd.DataFrame, config: NLPConfig | None = None) -> pd.DataFrame:
    """Aplica o pipeline configuravel sobre a coluna de texto do DataFrame."""
    config = config or obter_configuracao_nlp_padrao()
    validar_dataframe_entrada(df, config)

    logger.info("Iniciando pre-processamento de %s registros.", len(df))
    logger.info("Etapas ativas: %s", ", ".join(obter_etapas_ativas(config)))

    resources = carregar_recursos_nlp(config)
    processed_df = df.copy()
    processed_df[config.original_column] = processed_df[config.source_column].fillna("").astype(str)

    processed_records = [
        process_text(text, config=config, resources=resources)
        for text in processed_df[config.original_column].tolist()
    ]

    processed_df[config.processed_column] = [item["processed_text"] for item in processed_records]
    processed_df[config.tokens_column] = [item["tokens"] for item in processed_records]

    logger.info(
        "Pre-processamento concluido com backend '%s' e coluna final '%s'.",
        resources.backend,
        config.processed_column,
    )
    return processed_df


def process_text(text: str, config: NLPConfig, resources: NLPResources) -> dict:
    """Executa o pipeline sobre um unico texto e retorna saida estruturada."""
    working_text = str(text or "")

    if config.normalize_text:
        working_text = normalize_text(working_text)

    if config.clean_whitespace:
        working_text = clean_whitespace(working_text)

    if resources.nlp is not None:
        tokens = tokenize_with_spacy(working_text, config, resources)
    else:
        tokens = tokenize_with_fallback(working_text, config, resources)

    processed_text = " ".join(tokens)

    if config.clean_whitespace:
        processed_text = clean_whitespace(processed_text)

    return {
        "processed_text": processed_text,
        "tokens": tokens,
    }


def normalize_text(text: str) -> str:
    """Normaliza caracteres Unicode e padroniza quebras de linha."""
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def clean_whitespace(text: str) -> str:
    """Remove excessos de espaco para estabilizar a entrada textual."""
    return MULTISPACE_PATTERN.sub(" ", text.replace("\n", " ")).strip()


def carregar_recursos_nlp(config: NLPConfig) -> NLPResources:
    """Carrega spaCy quando disponivel e aplica fallback seguro quando necessario."""
    if not config.use_spacy:
        logger.info("spaCy desabilitado por configuracao. Usando fallback leve.")
        return NLPResources(
            nlp=None,
            stopwords=FALLBACK_STOPWORDS_PT,
            backend="fallback",
            lemma_available=False,
        )

    try:
        import spacy
    except ImportError:
        logger.warning("spaCy nao esta instalado. Usando fallback leve para tokenizacao.")
        return NLPResources(
            nlp=None,
            stopwords=FALLBACK_STOPWORDS_PT,
            backend="fallback",
            lemma_available=False,
        )

    try:
        nlp = spacy.load(config.spacy_model, disable=["ner", "parser"])
        backend = config.spacy_model
        logger.info("Modelo spaCy carregado: %s", config.spacy_model)
    except OSError:
        logger.warning(
            "Modelo spaCy '%s' indisponivel. Usando tokenizer spaCy em branco como fallback.",
            config.spacy_model,
        )
        nlp = spacy.blank(config.language)
        backend = f"spacy-blank:{config.language}"

    stopwords = set(getattr(nlp.Defaults, "stop_words", set()) or FALLBACK_STOPWORDS_PT)
    lemma_available = bool(getattr(nlp, "pipe_names", [])) and "lemmatizer" in nlp.pipe_names

    if config.lemmatize and not lemma_available:
        logger.warning("Lemmatizacao solicitada, mas indisponivel no backend atual. Etapa sera ignorada.")

    return NLPResources(
        nlp=nlp,
        stopwords=stopwords,
        backend=backend,
        lemma_available=lemma_available,
    )


def tokenize_with_spacy(text: str, config: NLPConfig, resources: NLPResources) -> list[str]:
    """Tokeniza com spaCy e aplica filtros opcionais de NLP."""
    nlp = cast(Any, resources.nlp)
    doc = nlp(text)
    tokens: list[str] = []

    for token in doc:
        if token.is_space:
            continue
        if config.remove_punctuation and token.is_punct:
            continue

        candidate = token.text
        if config.lemmatize and resources.lemma_available and token.lemma_:
            candidate = token.lemma_

        if config.lowercase:
            candidate = candidate.lower()

        candidate = candidate.strip()

        if not candidate:
            continue
        if config.remove_stopwords and candidate in resources.stopwords:
            continue

        tokens.append(candidate)

    return tokens


def tokenize_with_fallback(text: str, config: NLPConfig, resources: NLPResources) -> list[str]:
    """Tokeniza sem spaCy, preservando um fluxo leve e previsivel."""
    raw_tokens = TOKEN_PATTERN.findall(text)
    tokens: list[str] = []

    for token in raw_tokens:
        candidate = token.strip()
        if not candidate:
            continue

        if config.remove_punctuation and is_punctuation_token(candidate):
            continue

        if config.lowercase:
            candidate = candidate.lower()

        if config.remove_stopwords and candidate in resources.stopwords:
            continue

        if config.remove_punctuation:
            candidate = candidate.translate(PUNCTUATION_TABLE).strip()
            if not candidate:
                continue

        tokens.append(candidate)

    return tokens


def is_punctuation_token(token: str) -> bool:
    """Indica se um token e composto apenas por sinais de pontuacao."""
    return all(character in string.punctuation for character in token)


def validar_dataframe_entrada(df: pd.DataFrame, config: NLPConfig) -> None:
    """Valida se o DataFrame possui a coluna necessaria para NLP."""
    if config.source_column not in df.columns:
        raise ValueError(
            f"O DataFrame informado nao possui a coluna '{config.source_column}' para processamento."
        )


def obter_etapas_ativas(config: NLPConfig) -> list[str]:
    """Lista as etapas aplicadas no pipeline para fins de log e auditoria leve."""
    steps = []

    if config.normalize_text:
        steps.append("normalizacao")
    if config.clean_whitespace:
        steps.append("limpeza_espacos")
    if config.lowercase:
        steps.append("lowercase")
    if config.remove_punctuation:
        steps.append("remocao_pontuacao")
    if config.remove_stopwords:
        steps.append("remocao_stopwords")
    if config.lemmatize:
        steps.append("lematizacao")

    steps.append("tokenizacao")
    return steps
