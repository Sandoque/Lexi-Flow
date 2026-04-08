"""Camada de roteamento operacional assistido para a inferência do LexiFlow."""

from __future__ import annotations

from typing import Any


def definir_fluxo_operacional(
    macro_confidence: float,
    ambiguous_case: bool,
    priority: str,
    provider: str,
    genai_status: str,
) -> dict[str, Any]:
    """Traduz sinais do baseline e da GenAI em um fluxo operacional recomendável."""
    confidence_level = classificar_nivel_confianca(macro_confidence)
    normalized_priority = str(priority).strip().lower()

    if provider == "indisponível" or genai_status != "ok":
        return {
            "decision": "classificação assistida",
            "confidence_level": confidence_level,
            "reason": "A previsão baseline está disponível, mas o refinamento generativo não ficou utilizável nesta execução.",
            "action": "Registrar a macro e validar a classe detalhada com apoio humano antes da conclusão.",
            "operator_note": "Priorize uma revisão curta do caso e, no futuro, registre o feedback humano para retroalimentar o fluxo.",
            "review_required": True,
            "queue": "assistida",
            "badge_tone": "warning",
        }

    if ambiguous_case or confidence_level == "baixa":
        return {
            "decision": "revisão humana",
            "confidence_level": confidence_level,
            "reason": "A combinação de baixa confiança ou ambiguidade aumenta o risco de erro operacional.",
            "action": "Enviar o caso para triagem humana antes de confirmar a classificação final.",
            "operator_note": "Use a justificativa da GenAI como apoio, mas trate a decisão final como responsabilidade do operador.",
            "review_required": True,
            "queue": "humana",
            "badge_tone": "danger",
        }

    if confidence_level == "media":
        return {
            "decision": "classificação assistida",
            "confidence_level": confidence_level,
            "reason": "A previsão está consistente, mas ainda não é forte o suficiente para automação plena.",
            "action": "Permitir classificação assistida com confirmação rápida do operador.",
            "operator_note": "Confirme a classe detalhada sugerida e sinalize divergências para futura captura de feedback.",
            "review_required": True,
            "queue": "assistida",
            "badge_tone": "warning",
        }

    if normalized_priority == "alta":
        return {
            "decision": "classificação automática",
            "confidence_level": confidence_level,
            "reason": "Confiança alta com prioridade elevada favorece automação com monitoramento leve.",
            "action": "Registrar automaticamente a classificação e encaminhar para a fila prioritária.",
            "operator_note": "Monitore exceções e capture eventuais correções para ajustar limites de roteamento no futuro.",
            "review_required": False,
            "queue": "prioritária",
            "badge_tone": "success",
        }

    return {
        "decision": "classificação automática",
        "confidence_level": confidence_level,
        "reason": "Confiança alta e ausência de ambiguidade permitem uma execução mais direta.",
        "action": "Registrar automaticamente a classificação e seguir o fluxo operacional padrão.",
        "operator_note": "Mantenha amostragem de auditoria para futura camada de feedback humano.",
        "review_required": False,
        "queue": "automática",
        "badge_tone": "success",
    }


def classificar_nivel_confianca(confidence: float) -> str:
    """Classifica o score da macro em uma faixa operacional simples."""
    if confidence >= 0.75:
        return "alta"
    if confidence >= 0.45:
        return "media"
    return "baixa"


def obter_legenda_fluxo_operacional() -> list[dict[str, str]]:
    """Retorna uma legenda curta para a interface da camada operacional."""
    return [
        {
            "level": "Alta confiança",
            "description": "Permite classificação automática quando não há sinais fortes de ambiguidade.",
        },
        {
            "level": "Média confiança",
            "description": "Direciona para classificação assistida com confirmação rápida do operador.",
        },
        {
            "level": "Baixa confiança",
            "description": "Encaminha para revisão humana antes da decisão final.",
        },
    ]
