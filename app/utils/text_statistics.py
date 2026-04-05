"""Utilitarios para estatisticas textuais basicas."""

from __future__ import annotations

import pandas as pd


def calcular_estatisticas_textuais(text_series: pd.Series) -> dict:
    """Calcula metricas de tamanho dos textos para a EDA."""
    normalized = text_series.fillna("").astype(str).str.strip()
    char_counts = normalized.str.len()
    word_counts = normalized.str.split().str.len()

    shortest_index = char_counts.idxmin()
    longest_index = char_counts.idxmax()

    return {
        "avg_chars": round(float(char_counts.mean()), 2),
        "avg_words": round(float(word_counts.mean()), 2),
        "shortest_text": {
            "size": int(char_counts.loc[shortest_index]),
            "preview": resumir_texto(normalized.loc[shortest_index]),
        },
        "longest_text": {
            "size": int(char_counts.loc[longest_index]),
            "preview": resumir_texto(normalized.loc[longest_index]),
        },
    }


def resumir_texto(text: str, limit: int = 180) -> str:
    """Reduz um texto longo para apresentacao em cards e tabelas."""
    clean_text = " ".join(str(text).split())

    if len(clean_text) <= limit:
        return clean_text or "-"

    return f"{clean_text[:limit].rstrip()}..."
