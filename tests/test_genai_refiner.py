"""Testes da camada GenAI com foco em Groq e fallback para mock."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.genai_refiner import (
    FewShotExample,
    GenAIRefiner,
    GenAIRefinerError,
    GenAISettings,
    get_genai_settings_from_config,
    resolve_effective_provider_settings,
)


def build_settings(**overrides: object) -> GenAISettings:
    """Cria configuracoes padrao para testes unitarios da camada GenAI."""
    settings = {
        "requested_provider": "mock",
        "effective_provider": "mock",
        "model": "mock-model",
        "api_key": None,
        "api_key_source": None,
        "base_url": None,
        "temperature": 0.1,
        "timeout_seconds": 30,
        "mock_mode": True,
    }
    settings.update(overrides)
    return GenAISettings(**settings)


def test_resolve_effective_provider_settings_prefers_groq_api_key():
    """Garante fallback de GROQ_API_KEY quando GENAI_API_KEY nao foi definido."""
    config = {
        "GENAI_API_KEY": "",
        "GROQ_API_KEY": "groq-secret",
        "GENAI_BASE_URL": "",
    }

    resolved = resolve_effective_provider_settings(config, provider="groq", mock_mode=False)

    assert resolved["effective_provider"] == "groq"
    assert resolved["api_key"] == "groq-secret"
    assert resolved["api_key_source"] == "GROQ_API_KEY"
    assert resolved["base_url"] == "https://api.groq.com/openai/v1"


def test_get_genai_settings_from_config_keeps_groq_base_url_and_provider():
    """Garante que as configuracoes finais preservam o provider Groq."""
    config = {
        "GENAI_PROVIDER": "groq",
        "GENAI_MOCK_MODE": False,
        "GENAI_MODEL": "llama-3.3-70b-versatile",
        "GENAI_API_KEY": "",
        "GROQ_API_KEY": "groq-secret",
        "GENAI_BASE_URL": "",
        "GENAI_TEMPERATURE": 0.2,
        "GENAI_TIMEOUT_SECONDS": 45,
    }

    settings = get_genai_settings_from_config(config)

    assert settings.requested_provider == "groq"
    assert settings.effective_provider == "groq"
    assert settings.api_key == "groq-secret"
    assert settings.api_key_source == "GROQ_API_KEY"
    assert settings.base_url == "https://api.groq.com/openai/v1"
    assert settings.mock_mode is False


def test_refine_raises_friendly_error_when_groq_has_no_key():
    """Garante mensagem amigavel quando Groq foi solicitado sem credencial."""
    refiner = GenAIRefiner(
        build_settings(
            requested_provider="groq",
            effective_provider="groq",
            model="llama-3.3-70b-versatile",
            mock_mode=False,
        )
    )

    with pytest.raises(GenAIRefinerError) as exc_info:
        refiner.refine(
            text="Preciso da segunda via do boleto.",
            predicted_macro="Financeiro",
            valid_detailed_classes=["Boleto", "Cobranca"],
        )

    assert "Groq" in str(exc_info.value)
    assert "GROQ_API_KEY" in str(exc_info.value)


def test_refine_falls_back_to_mock_on_timeout(monkeypatch: pytest.MonkeyPatch):
    """Garante fallback automatico para mock quando a chamada remota expira."""

    class FakeTimeoutError(Exception):
        """Simula timeout do cliente OpenAI-compatible."""

    class FakeCompletions:
        def create(self, **_: object) -> object:
            raise FakeTimeoutError("timeout")

    class FakeOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai_module = SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIConnectionError=RuntimeError,
        APITimeoutError=FakeTimeoutError,
        AuthenticationError=PermissionError,
    )

    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)

    refiner = GenAIRefiner(
        build_settings(
            requested_provider="groq",
            effective_provider="groq",
            model="llama-3.3-70b-versatile",
            api_key="groq-secret",
            api_key_source="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1",
            mock_mode=False,
        )
    )

    result = refiner.refine(
        text="Preciso da segunda via do boleto com urgencia.",
        predicted_macro="Financeiro",
        valid_detailed_classes=["Boleto", "Cobranca", "Pagamento"],
        few_shot_examples=[
            FewShotExample(
                text="Emitir segunda via de boleto.",
                macro_class="Financeiro",
                detailed_class="Boleto",
                justification="Caso financeiro ligado a boleto.",
            )
        ],
    )

    assert result["requested_provider"] == "groq"
    assert result["provider"] == "mock"
    assert result["mode"] == "mock"
    assert result["few_shot_count"] == 1
    assert result["fallback_reason"] == "timeout na chamada do provider"
    assert result["result"]["detailed_class"] in {"Boleto", "Cobranca", "Pagamento"}
