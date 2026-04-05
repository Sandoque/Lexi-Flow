"""Camada GenAI para refinamento de classe detalhada com provider configuravel."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.services.baseline_classifier import BaselineError, carregar_artefatos_baseline

logger = logging.getLogger(__name__)


class GenAIRefinerError(Exception):
    """Representa erros esperados durante o refinamento com IA generativa."""


@dataclass(slots=True)
class FewShotExample:
    """Representa um exemplo curto para few-shot prompting."""

    text: str
    macro_class: str
    detailed_class: str
    justification: str
    priority: str = "media"
    ambiguous: bool = False


@dataclass(slots=True)
class GenAISettings:
    """Configuracoes operacionais da camada GenAI."""

    requested_provider: str
    effective_provider: str
    model: str
    api_key: str | None
    api_key_source: str | None
    base_url: str | None
    temperature: float
    timeout_seconds: int
    mock_mode: bool


class GenAIRefiner:
    """Orquestra prompt, provider e parsing do refinamento detalhado."""

    def __init__(self, settings: GenAISettings):
        self.settings = settings
        logger.info(
            "GenAI inicializado com provider solicitado='%s' e provider efetivo='%s'.",
            settings.requested_provider,
            settings.effective_provider,
        )
        self.provider = build_provider(settings)

    def refine(
        self,
        text: str,
        predicted_macro: str,
        valid_detailed_classes: list[str],
        few_shot_examples: list[FewShotExample] | None = None,
    ) -> dict:
        """Executa o refinamento da classe detalhada com retorno estruturado."""
        if not text or not text.strip():
            raise GenAIRefinerError("Informe um texto para classificacao.")
        if not predicted_macro or not predicted_macro.strip():
            raise GenAIRefinerError("Informe a macroclasse prevista para o refinamento.")
        if not valid_detailed_classes:
            raise GenAIRefinerError("Nao ha classes detalhadas validas para a macroclasse informada.")

        prompt = build_structured_prompt(
            text=text,
            predicted_macro=predicted_macro,
            valid_detailed_classes=valid_detailed_classes,
            few_shot_examples=few_shot_examples or [],
        )
        raw_response = self.provider.generate_structured_completion(prompt)
        parsed = parse_refiner_response(raw_response, valid_detailed_classes)

        return {
            "provider": self.provider.active_provider,
            "requested_provider": self.settings.requested_provider,
            "api_key_source": self.settings.api_key_source,
            "model": self.settings.model,
            "mode": self.provider.active_mode,
            "fallback_reason": self.provider.fallback_reason,
            "prompt": prompt,
            "result": parsed,
        }


class BaseGenAIProvider:
    """Contrato minimo dos providers suportados pela camada GenAI."""

    def __init__(self, settings: GenAISettings):
        self.settings = settings
        self.active_provider = settings.effective_provider
        self.active_mode = "mock" if settings.mock_mode else "api"
        self.fallback_reason: str | None = None

    def generate_structured_completion(self, prompt: str) -> str:
        """Gera resposta textual estruturada a partir do prompt."""
        raise NotImplementedError


class MockGenAIProvider(BaseGenAIProvider):
    """Provider fake para demonstracao local e desenvolvimento sem custo."""

    def generate_structured_completion(self, prompt: str) -> str:
        self.active_provider = "mock"
        self.active_mode = "mock"
        payload = extract_payload_from_prompt(prompt)
        valid_classes = payload["valid_detailed_classes"]
        text = payload["text"].lower()

        selected = choose_mock_class(text, valid_classes)
        ambiguous = is_ambiguous_case(text, valid_classes)
        priority = infer_priority(text)
        justification = (
            "Sugestao mock baseada em correspondencia lexical simples, pensada apenas para demonstracao operacional."
        )

        return json.dumps(
            {
                "detailed_class": selected,
                "justification": justification,
                "priority": priority,
                "ambiguous_case": ambiguous,
            },
            ensure_ascii=False,
        )


class OpenAICompatibleProvider(BaseGenAIProvider):
    """Provider generico para APIs compatíveis com o cliente OpenAI."""

    def generate_structured_completion(self, prompt: str) -> str:
        self.active_provider = self.settings.effective_provider
        self.active_mode = "api"
        self.fallback_reason = None
        if not self.settings.api_key:
            raise GenAIRefinerError(
                build_missing_api_key_message(self.settings)
            )

        try:
            from openai import OpenAI
            from openai import APIConnectionError, APITimeoutError, AuthenticationError
        except ImportError as exc:
            raise GenAIRefinerError("O pacote openai nao esta disponivel no ambiente atual.") from exc

        client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or None,
            timeout=self.settings.timeout_seconds,
        )
        try:
            response = client.chat.completions.create(
                model=self.settings.model,
                temperature=self.settings.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Voce e um assistente de classificacao textual operacional. "
                            "Responda somente com JSON valido."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            logger.warning(
                "Falha de autenticacao no provider '%s'.",
                self.settings.effective_provider,
            )
            raise GenAIRefinerError(
                "Falha de autenticacao no provider configurado. Revise a API key e tente novamente."
            ) from exc
        except APITimeoutError as exc:
            logger.warning(
                "Timeout no provider '%s'. Tentando fallback para mock.",
                self.settings.effective_provider,
            )
            return self._fallback_to_mock(prompt, reason="timeout na chamada do provider")
        except APIConnectionError as exc:
            logger.warning(
                "Erro de rede no provider '%s'. Tentando fallback para mock.",
                self.settings.effective_provider,
            )
            return self._fallback_to_mock(prompt, reason="falha de rede na chamada do provider")
        except Exception as exc:
            logger.exception(
                "Erro inesperado no provider '%s'. Aplicando fallback para mock.",
                self.settings.effective_provider,
            )
            return self._fallback_to_mock(prompt, reason="erro inesperado na chamada do provider")

        content = response.choices[0].message.content
        if not content:
            raise GenAIRefinerError("O provider retornou uma resposta vazia.")
        return content

    def _fallback_to_mock(self, prompt: str, reason: str) -> str:
        """Aplica fallback controlado para mock sem quebrar a interface."""
        self.active_provider = "mock"
        self.active_mode = "mock"
        self.fallback_reason = reason
        return fallback_to_mock_response(
            settings=self.settings,
            prompt=prompt,
            reason=reason,
        )


def build_provider(settings: GenAISettings) -> BaseGenAIProvider:
    """Constroi o provider de acordo com a configuracao."""
    if settings.mock_mode or settings.effective_provider == "mock":
        return MockGenAIProvider(settings)

    if settings.effective_provider in {"openai", "groq", "openai_compatible"}:
        return OpenAICompatibleProvider(settings)

    raise GenAIRefinerError(f"Provider GenAI nao suportado: {settings.effective_provider}.")


def get_genai_settings_from_config(config: Any) -> GenAISettings:
    """Lê as configuracoes GenAI a partir da app config."""
    provider = str(config.get("GENAI_PROVIDER", "mock")).strip().lower()
    raw_mock_mode = config.get("GENAI_MOCK_MODE", True)
    if isinstance(raw_mock_mode, str):
        mock_mode = raw_mock_mode.strip().lower() in {"1", "true", "yes", "on"}
    else:
        mock_mode = bool(raw_mock_mode)
    resolved = resolve_effective_provider_settings(config, provider=provider, mock_mode=mock_mode)

    return GenAISettings(
        requested_provider=provider,
        effective_provider=resolved["effective_provider"],
        model=str(config.get("GENAI_MODEL", "mock-model")),
        api_key=resolved["api_key"],
        api_key_source=resolved["api_key_source"],
        base_url=resolved["base_url"],
        temperature=float(config.get("GENAI_TEMPERATURE", 0.1)),
        timeout_seconds=int(config.get("GENAI_TIMEOUT_SECONDS", 30)),
        mock_mode=resolved["mock_mode"],
    )


def resolve_effective_provider_settings(
    config: Any,
    provider: str,
    mock_mode: bool,
) -> dict[str, Any]:
    """Resolve provider efetivo, origem da chave e base URL sem expor segredos."""
    if mock_mode or provider == "mock":
        logger.info("GenAI configurado para usar modo mock.")
        return {
            "effective_provider": "mock",
            "api_key": None,
            "api_key_source": None,
            "base_url": None,
            "mock_mode": True,
        }

    if provider == "groq":
        genai_key = normalize_optional_string(config.get("GENAI_API_KEY"))
        groq_key = normalize_optional_string(config.get("GROQ_API_KEY"))
        base_url = normalize_optional_string(config.get("GENAI_BASE_URL")) or "https://api.groq.com/openai/v1"

        if genai_key:
            api_key = genai_key
            api_key_source = "GENAI_API_KEY"
        elif groq_key:
            api_key = groq_key
            api_key_source = "GROQ_API_KEY"
        else:
            api_key = None
            api_key_source = None

        logger.info(
            "GenAI configurado para usar Groq com base_url='%s' e origem de chave='%s'.",
            base_url,
            api_key_source or "nenhuma",
        )
        return {
            "effective_provider": "groq",
            "api_key": api_key,
            "api_key_source": api_key_source,
            "base_url": base_url,
            "mock_mode": False,
        }

    openai_key = normalize_optional_string(config.get("GENAI_API_KEY")) or normalize_optional_string(
        config.get("OPENAI_API_KEY")
    )
    api_key_source = "GENAI_API_KEY" if normalize_optional_string(config.get("GENAI_API_KEY")) else (
        "OPENAI_API_KEY" if normalize_optional_string(config.get("OPENAI_API_KEY")) else None
    )
    base_url = normalize_optional_string(config.get("GENAI_BASE_URL"))

    logger.info(
        "GenAI configurado para usar provider '%s' com origem de chave='%s'.",
        provider,
        api_key_source or "nenhuma",
    )
    return {
        "effective_provider": provider,
        "api_key": openai_key,
        "api_key_source": api_key_source,
        "base_url": base_url,
        "mock_mode": False,
    }


def build_structured_prompt(
    text: str,
    predicted_macro: str,
    valid_detailed_classes: list[str],
    few_shot_examples: list[FewShotExample],
) -> str:
    """Monta um prompt estruturado e pronto para few-shot contextual."""
    payload = {
        "task": "Refinar a classe detalhada de um texto sem substituir o baseline.",
        "baseline_role": (
            "O baseline ja definiu a macroclasse. Sua funcao e sugerir a classe detalhada dentro das opcoes permitidas."
        ),
        "instructions": [
            "Escolha somente uma classe detalhada valida.",
            "Responda em JSON com os campos: detailed_class, justification, priority, ambiguous_case.",
            "Mantenha a justificativa curta e operacional.",
            "Use a prioridade apenas se o texto indicar urgencia, criticidade ou impacto operacional.",
            "Marque ambiguous_case como true quando houver sinais conflitantes ou mais de uma classe plausivel.",
        ],
        "predicted_macro": predicted_macro,
        "valid_detailed_classes": valid_detailed_classes,
        "few_shot_examples": [
            {
                "text": example.text,
                "macro_class": example.macro_class,
                "detailed_class": example.detailed_class,
                "justification": example.justification,
                "priority": example.priority,
                "ambiguous_case": example.ambiguous,
            }
            for example in few_shot_examples
        ],
        "text": text,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_refiner_response(raw_response: str, valid_detailed_classes: list[str]) -> dict:
    """Valida e normaliza a resposta estruturada do provider."""
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise GenAIRefinerError("A resposta do provider nao veio em JSON valido.") from exc

    detailed_class = str(data.get("detailed_class", "")).strip()
    if detailed_class not in valid_detailed_classes:
        raise GenAIRefinerError("A classe detalhada sugerida nao pertence ao conjunto permitido para a macroclasse.")

    justification = str(data.get("justification", "")).strip() or "Sem justificativa informada."
    priority = str(data.get("priority", "")).strip() or "nao definida"
    ambiguous_case = bool(data.get("ambiguous_case", False))

    return {
        "detailed_class": detailed_class,
        "justification": justification,
        "priority": priority,
        "ambiguous_case": ambiguous_case,
    }


def get_demo_macro_options(artifacts_folder: str | None = None) -> list[dict]:
    """Retorna as opcoes de macro e classes detalhadas para a tela demo."""
    if artifacts_folder:
        try:
            artifacts = carregar_artefatos_baseline(artifacts_folder)
            return artifacts["metadata"]["refinement_context"]["macro_detail_options"]
        except (BaselineError, KeyError):
            logger.info("Artefatos hierarquicos indisponiveis. Usando opcoes demo padrao para GenAI.")

    return [
        {
            "macro_class": "Suporte",
            "detail_options": ["Login", "Senha", "Anexo", "Acesso"],
            "detail_count": 4,
        },
        {
            "macro_class": "Financeiro",
            "detail_options": ["Boleto", "Cobranca", "Pagamento", "Reembolso"],
            "detail_count": 4,
        },
        {
            "macro_class": "Cadastro",
            "detail_options": ["Atualizacao", "Endereco", "Dados pessoais", "Contato"],
            "detail_count": 4,
        },
    ]


def get_demo_few_shot_examples() -> list[FewShotExample]:
    """Retorna exemplos simples para few-shot prompting na demonstracao."""
    return [
        FewShotExample(
            text="Nao consigo acessar a conta porque a senha expirou.",
            macro_class="Suporte",
            detailed_class="Senha",
            justification="O texto aponta diretamente para recuperacao ou expiracao de senha.",
            priority="alta",
            ambiguous=False,
        ),
        FewShotExample(
            text="Preciso emitir a segunda via do boleto vencido.",
            macro_class="Financeiro",
            detailed_class="Boleto",
            justification="A mencao explicita a segunda via de boleto dentro da macro financeira.",
            priority="media",
            ambiguous=False,
        ),
        FewShotExample(
            text="Quero alterar meu endereco cadastrado no sistema.",
            macro_class="Cadastro",
            detailed_class="Endereco",
            justification="O pedido trata de atualizacao cadastral especificamente ligada a endereco.",
            priority="baixa",
            ambiguous=False,
        ),
    ]


def extract_payload_from_prompt(prompt: str) -> dict:
    """Recupera o payload estruturado do prompt para uso no modo mock."""
    try:
        return json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise GenAIRefinerError("Nao foi possivel interpretar o prompt estruturado.") from exc


def fallback_to_mock_response(settings: GenAISettings, prompt: str, reason: str) -> str:
    """Executa fallback seguro para mock quando o provider remoto falha."""
    logger.warning(
        "Fallback automatico para mock ativado. Provider solicitado='%s', motivo='%s'.",
        settings.requested_provider,
        reason,
    )
    mock_settings = GenAISettings(
        requested_provider=settings.requested_provider,
        effective_provider="mock",
        model="mock-fallback",
        api_key=None,
        api_key_source=None,
        base_url=None,
        temperature=settings.temperature,
        timeout_seconds=settings.timeout_seconds,
        mock_mode=True,
    )
    provider = MockGenAIProvider(mock_settings)
    return provider.generate_structured_completion(prompt)


def build_missing_api_key_message(settings: GenAISettings) -> str:
    """Monta uma mensagem amigavel para ausencia de credencial."""
    if settings.effective_provider == "groq":
        return (
            "Nenhuma chave foi configurada para Groq. Defina GENAI_API_KEY ou GROQ_API_KEY em secret.env "
            "ou nas variaveis do sistema."
        )
    return (
        "Nenhuma chave foi configurada para o provider GenAI. Defina GENAI_API_KEY "
        "ou a chave especifica do provider."
    )


def normalize_optional_string(value: Any) -> str | None:
    """Normaliza strings opcionais vindas da configuracao."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def choose_mock_class(text: str, valid_classes: list[str]) -> str:
    """Seleciona uma classe detalhada mock com base em pistas lexicais simples."""
    normalized_options = {option.lower(): option for option in valid_classes}

    keyword_map = {
        "login": ["login", "acesso", "entrar", "autenticação", "autenticacao"],
        "senha": ["senha", "reset", "recuperar", "expirada"],
        "anexo": ["anexo", "arquivo", "documento", "upload"],
        "acesso": ["acesso", "conta", "perfil", "permissão", "permissao"],
        "boleto": ["boleto", "segunda via", "2 via"],
        "cobranca": ["cobrança", "cobranca", "fatura", "debito"],
        "pagamento": ["pagamento", "pagar", "quitado"],
        "reembolso": ["reembolso", "estorno", "devolução", "devolucao"],
        "atualizacao": ["atualizar", "atualização", "atualizacao", "cadastro"],
        "endereco": ["endereço", "endereco", "logradouro", "cep"],
        "dados pessoais": ["cpf", "nome", "dados pessoais", "data de nascimento"],
        "contato": ["telefone", "email", "contato", "celular"],
    }

    for option_lower, original in normalized_options.items():
        for keyword in keyword_map.get(option_lower, []):
            if keyword in text:
                return original

    return valid_classes[0]


def infer_priority(text: str) -> str:
    """Infere uma prioridade simples para a demonstracao mock."""
    high_terms = ["urgente", "critico", "bloqueado", "parado", "imediato"]
    medium_terms = ["hoje", "prazo", "vencido", "atraso"]

    if any(term in text for term in high_terms):
        return "alta"
    if any(term in text for term in medium_terms):
        return "media"
    return "baixa"


def is_ambiguous_case(text: str, valid_classes: list[str]) -> bool:
    """Sinaliza ambiguidade quando o texto parece compatível com multiplas classes."""
    matches = 0
    for option in valid_classes:
        if option.lower() in text:
            matches += 1
    return matches > 1 or "ou" in text
