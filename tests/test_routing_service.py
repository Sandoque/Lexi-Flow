"""Testes da camada de roteamento operacional do LexiFlow."""

from app.services.routing_service import definir_fluxo_operacional


def test_routing_service_returns_automatic_for_high_confidence():
    """Alta confianca sem ambiguidade deve permitir automacao."""
    result = definir_fluxo_operacional(
        macro_confidence=0.91,
        ambiguous_case=False,
        priority="media",
        provider="mock",
        genai_status="ok",
    )

    assert result["decision"] == "classificacao automatica"
    assert result["confidence_level"] == "alta"
    assert result["review_required"] is False


def test_routing_service_returns_assisted_for_medium_confidence():
    """Media confianca deve gerar classificacao assistida."""
    result = definir_fluxo_operacional(
        macro_confidence=0.58,
        ambiguous_case=False,
        priority="media",
        provider="groq",
        genai_status="ok",
    )

    assert result["decision"] == "classificacao assistida"
    assert result["confidence_level"] == "media"
    assert result["queue"] == "assistida"


def test_routing_service_returns_human_review_for_low_confidence():
    """Baixa confianca deve escalar para revisao humana."""
    result = definir_fluxo_operacional(
        macro_confidence=0.22,
        ambiguous_case=False,
        priority="baixa",
        provider="groq",
        genai_status="ok",
    )

    assert result["decision"] == "revisao humana"
    assert result["confidence_level"] == "baixa"
    assert result["review_required"] is True
