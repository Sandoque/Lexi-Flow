"""Camada de roteamento operacional assistido para a inferencia do LexiFlow."""

from __future__ import annotations

from typing import Any


def definir_fluxo_operacional(
    macro_confidence: float,
    ambiguous_case: bool,
    priority: str,
    provider: str,
    genai_status: str,
) -> dict[str, Any]:
    """Traduz sinais do baseline e da GenAI em um fluxo operacional recomendavel."""
    confidence_level = classificar_nivel_confianca(macro_confidence)
    normalized_priority = str(priority).strip().lower()

    if provider == "indisponivel" or genai_status != "ok":
        return {
            "decision": "classificacao assistida",
            "confidence_level": confidence_level,
            "reason": "A previsao baseline esta disponivel, mas o refinamento generativo nao ficou utilizavel nesta execucao.",
            "action": "Registrar a macro e validar a classe detalhada com apoio humano antes da conclusao.",
            "operator_note": "Priorize uma revisao curta do caso e, no futuro, registre o feedback humano para retroalimentar o fluxo.",
            "review_required": True,
            "queue": "assistida",
            "badge_tone": "warning",
        }

    if ambiguous_case or confidence_level == "baixa":
        return {
            "decision": "revisao humana",
            "confidence_level": confidence_level,
            "reason": "A combinacao de baixa confianca ou ambiguidade aumenta o risco de erro operacional.",
            "action": "Enviar o caso para triagem humana antes de confirmar a classificacao final.",
            "operator_note": "Use a justificativa da GenAI como apoio, mas trate a decisao final como responsabilidade do operador.",
            "review_required": True,
            "queue": "humana",
            "badge_tone": "danger",
        }

    if confidence_level == "media":
        return {
            "decision": "classificacao assistida",
            "confidence_level": confidence_level,
            "reason": "A previsao esta consistente, mas ainda nao e forte o suficiente para automacao plena.",
            "action": "Permitir classificacao assistida com confirmacao rapida do operador.",
            "operator_note": "Confirme a classe detalhada sugerida e sinalize divergencias para futura captura de feedback.",
            "review_required": True,
            "queue": "assistida",
            "badge_tone": "warning",
        }

    if normalized_priority == "alta":
        return {
            "decision": "classificacao automatica",
            "confidence_level": confidence_level,
            "reason": "Confianca alta com prioridade elevada favorece automacao com monitoramento leve.",
            "action": "Registrar automaticamente a classificacao e encaminhar para a fila prioritaria.",
            "operator_note": "Monitore excecoes e capture eventuais correcoes para ajustar limites de roteamento no futuro.",
            "review_required": False,
            "queue": "prioritaria",
            "badge_tone": "success",
        }

    return {
        "decision": "classificacao automatica",
        "confidence_level": confidence_level,
        "reason": "Confianca alta e ausencia de ambiguidade permitem uma execucao mais direta.",
        "action": "Registrar automaticamente a classificacao e seguir o fluxo operacional padrao.",
        "operator_note": "Mantenha amostragem de auditoria para futura camada de feedback humano.",
        "review_required": False,
        "queue": "automatica",
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
            "level": "Alta confianca",
            "description": "Permite classificacao automatica quando nao ha sinais fortes de ambiguidade.",
        },
        {
            "level": "Media confianca",
            "description": "Direciona para classificacao assistida com confirmacao rapida do operador.",
        },
        {
            "level": "Baixa confianca",
            "description": "Encaminha para revisao humana antes da decisao final.",
        },
    ]
