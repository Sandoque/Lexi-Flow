"""Testes unitarios da camada de pre-processamento textual."""

from typing import cast

import pandas as pd

from app.services.nlp_config import NLPConfig
from app.services.preprocessing_service import clean_whitespace, normalize_text, prepare_texts


def test_prepare_texts_preserva_texto_original_e_gera_colunas_processadas():
    """Garante que o pipeline cria colunas derivadas sem perder o texto original."""
    df = pd.DataFrame(
        {
            "texto": [
                "  Olá, Mundo!   Este é um TESTE.  ",
                "Canal: E-mail; assunto: cobrança.",
            ]
        }
    )
    config = NLPConfig(use_spacy=False, remove_stopwords=False, lemmatize=False)

    result = prepare_texts(df, config)
    texto_original = cast(str, result.loc[0, "texto_original"])
    texto_processado = cast(str, result.loc[0, "texto_processado"])
    texto_tokens = cast(list[str], result.loc[1, "texto_tokens"])

    assert "texto_original" in result.columns
    assert "texto_processado" in result.columns
    assert "texto_tokens" in result.columns
    assert texto_original.startswith("  Olá")
    assert texto_processado == "olá mundo este é um teste"
    assert texto_tokens == ["canal", "e", "mail", "assunto", "cobrança"]


def test_prepare_texts_remove_stopwords_quando_habilitado():
    """Garante que o pipeline remove stopwords no modo configurado."""
    df = pd.DataFrame({"texto": ["Esse é um texto de teste para a base."]})
    config = NLPConfig(use_spacy=False, remove_stopwords=True, lemmatize=False)

    result = prepare_texts(df, config)
    texto_processado = cast(str, result.loc[0, "texto_processado"])

    assert texto_processado == "esse é texto teste base"


def test_prepare_texts_pode_manter_pontuacao_quando_configurado():
    """Garante que a pontuacao pode ser preservada por configuracao."""
    df = pd.DataFrame({"texto": ["Teste, com pontuação!"]})
    config = NLPConfig(use_spacy=False, remove_punctuation=False, remove_stopwords=False)

    result = prepare_texts(df, config)
    texto_tokens = cast(list[str], result.loc[0, "texto_tokens"])
    texto_processado = cast(str, result.loc[0, "texto_processado"])

    assert texto_tokens == ["teste", ",", "com", "pontuação", "!"]
    assert texto_processado == "teste , com pontuação !"


def test_funcoes_basicas_de_limpeza_textual():
    """Garante comportamento previsivel das funcoes utilitarias centrais."""
    assert normalize_text("Texto\u00a0com espaço") == "Texto com espaço"
    assert clean_whitespace(" linha 1 \n\n linha 2  ") == "linha 1 linha 2"
