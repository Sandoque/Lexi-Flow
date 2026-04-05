"""Utilitarios para gerar graficos leves para a interface web."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gerar_grafico_barras_base64(
    series: pd.Series,
    title: str,
    color: str,
    horizontal: bool = False,
) -> str | None:
    """Gera um grafico de barras em memoria e retorna um data URI."""
    if series.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if horizontal:
        ax.barh(series.index.astype(str), series.values, color=color)
        ax.invert_yaxis()
    else:
        ax.bar(series.index.astype(str), series.values, color=color)
        ax.tick_params(axis="x", rotation=25)

    ax.set_title(title, fontsize=12, pad=14, color="#152033")
    ax.grid(axis="y" if not horizontal else "x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c7d3e4")
    ax.tick_params(colors="#5d687b")
    ax.set_axisbelow(True)

    buffer = BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def gerar_matriz_confusao_base64(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> str:
    """Gera uma matriz de confusao em memoria para exibicao na interface."""
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=12, pad=14, color="#152033")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            color = "#ffffff" if value > threshold else "#152033"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
