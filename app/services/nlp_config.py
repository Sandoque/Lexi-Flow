"""Configuracao reutilizavel para o pipeline de NLP do LexiFlow."""

from dataclasses import dataclass


@dataclass(slots=True)
class NLPConfig:
    """Define as etapas e colunas usadas no pre-processamento textual."""

    source_column: str = "texto"
    original_column: str = "texto_original"
    processed_column: str = "texto_processado"
    tokens_column: str = "texto_tokens"
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False
    normalize_text: bool = True
    clean_whitespace: bool = True
    use_spacy: bool = True
    language: str = "pt"
    spacy_model: str = "pt_core_news_sm"


def obter_configuracao_nlp_padrao() -> NLPConfig:
    """Retorna a configuracao base recomendada para o case."""
    return NLPConfig()
