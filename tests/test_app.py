"""Testes basicos de smoke para a aplicacao Flask do LexiFlow."""

from io import BytesIO
from pathlib import Path

from app import create_app
from app.services.genai_refiner import OpenAICompatibleProvider


def test_create_app_uses_testing_config():
    """Garante que a fabrica retorna uma instancia Flask configurada."""
    app = create_app("testing")

    assert app is not None
    assert app.config["TESTING"] is True


def test_home_route_returns_success():
    """Garante que a home principal responde com sucesso."""
    app = create_app("testing")
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"LexiFlow" in response.data


def test_health_route_returns_ok_payload():
    """Garante que a rota de healthcheck retorna status esperado."""
    app = create_app("testing")
    client = app.test_client()

    response = client.get("/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["application"] == "LexiFlow"


def test_upload_route_processes_valid_csv(tmp_path: Path):
    """Garante que a ingestao valida e resume um CSV valido."""
    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = tmp_path
    client = app.test_client()

    csv_content = (
        "id_registro,texto,canal_origem,data,classe_macro,classe_detalhada\n"
        "1,Texto de teste,email,2026-04-04,Atendimento,Consulta\n"
        "2,Outro texto,chat,2026-04-05,Suporte,Abertura\n"
    )

    response = client.post(
        "/upload",
        data={"dataset": (BytesIO(csv_content.encode("utf-8")), "base.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Arquivo validado e salvo com sucesso" in response.data
    assert b"Primeiras 10 linhas" in response.data
    assert (tmp_path / "base.csv").exists()


def test_upload_route_rejects_missing_columns(tmp_path: Path):
    """Garante que a ingestao rejeita datasets com schema incompleto."""
    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = tmp_path
    client = app.test_client()

    csv_content = "id_registro,texto\n1,Texto incompleto\n"

    response = client.post(
        "/upload",
        data={"dataset": (BytesIO(csv_content.encode("utf-8")), "invalido.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"colunas obrigatorias" in response.data


def test_eda_route_uses_latest_dataset(tmp_path: Path):
    """Garante que a EDA utiliza o CSV mais recente salvo em data/raw."""
    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = tmp_path
    client = app.test_client()

    csv_content = (
        "id_registro,texto,canal_origem,data,classe_macro,classe_detalhada\n"
        "1,Texto curto,email,2026-04-04,Atendimento,Consulta\n"
        "2,Texto um pouco maior para teste,chat,2026-04-05,Suporte,Abertura\n"
        "3,Outro registro para analise,email,2026-04-06,Atendimento,Retorno\n"
    )
    (tmp_path / "eda.csv").write_text(csv_content, encoding="utf-8")

    response = client.get("/eda")

    assert response.status_code == 200
    assert b"Painel inicial do dataset" in response.data
    assert b"Atendimento" in response.data
    assert b"Suporte" in response.data


def test_eda_route_redirects_when_no_dataset(tmp_path: Path):
    """Garante que a EDA redireciona para upload sem dataset disponivel."""
    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = tmp_path
    client = app.test_client()

    response = client.get("/eda", follow_redirects=True)

    assert response.status_code == 200
    assert b"Nenhum CSV disponivel para analise" in response.data
    assert b"Upload de CSV" in response.data


def test_baseline_route_trains_with_valid_dataset(tmp_path: Path):
    """Garante que a rota baseline hierarquica treina e renderiza os dois niveis."""
    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = tmp_path
    app.config["ARTIFACTS_FOLDER"] = tmp_path / "artifacts"
    client = app.test_client()

    csv_content = (
        "id_registro,texto,canal_origem,data,classe_macro,classe_detalhada\n"
        "1,Erro no login do sistema,email,2026-04-04,Suporte,Login\n"
        "2,Problema com senha esquecida,chat,2026-04-05,Suporte,Senha\n"
        "3,Duvida sobre segunda via de boleto,email,2026-04-06,Financeiro,Boleto\n"
        "4,Quero atualizar dados cadastrais,portal,2026-04-07,Cadastro,Atualizacao\n"
        "5,Erro ao anexar documento,chat,2026-04-08,Suporte,Anexo\n"
        "6,Consulta sobre cobranca em aberto,email,2026-04-09,Financeiro,Cobranca\n"
        "7,Pedido de alteracao de endereco,portal,2026-04-10,Cadastro,Endereco\n"
        "8,Problema no acesso a conta,chat,2026-04-11,Suporte,Login\n"
    )
    (tmp_path / "baseline.csv").write_text(csv_content, encoding="utf-8")

    response = client.get("/baseline")

    assert response.status_code == 200
    assert b"Painel supervisionado em dois niveis" in response.data
    assert b"Nivel 1" in response.data
    assert b"Nivel 2" in response.data
    assert b"Matriz de confusao" in response.data


def test_genai_demo_route_renders_with_mock_mode(tmp_path: Path):
    """Garante que a tela GenAI responde com provider mock por padrao."""
    app = create_app("testing")
    app.config["ARTIFACTS_FOLDER"] = tmp_path / "artifacts"
    app.config["GENAI_PROVIDER"] = "mock"
    app.config["GENAI_MOCK_MODE"] = True
    client = app.test_client()

    response = client.get("/genai-demo")

    assert response.status_code == 200
    assert b"GenAI Demo para refinamento detalhado" in response.data
    assert b"Classes detalhadas permitidas" in response.data


def test_genai_demo_route_classifies_text_in_mock_mode(tmp_path: Path):
    """Garante que a camada GenAI retorna uma resposta estruturada em modo mock."""
    app = create_app("testing")
    app.config["ARTIFACTS_FOLDER"] = tmp_path / "artifacts"
    app.config["GENAI_PROVIDER"] = "mock"
    app.config["GENAI_MOCK_MODE"] = True
    client = app.test_client()

    response = client.post(
        "/genai-demo",
        data={
            "macro_class": "Financeiro",
            "text_input": "Preciso da segunda via do boleto com urgencia hoje.",
        },
    )

    assert response.status_code == 200
    assert b"Sugestao do LLM" in response.data
    assert b"Boleto" in response.data
    assert b"mock" in response.data


def test_genai_demo_route_shows_friendly_error_when_groq_has_no_key(tmp_path: Path):
    """Garante mensagem amigavel quando Groq foi solicitado sem chave."""
    app = create_app("testing")
    app.config["ARTIFACTS_FOLDER"] = tmp_path / "artifacts"
    app.config["GENAI_PROVIDER"] = "groq"
    app.config["GENAI_MOCK_MODE"] = False
    app.config["GENAI_API_KEY"] = ""
    app.config["GROQ_API_KEY"] = ""
    client = app.test_client()

    response = client.post(
        "/genai-demo",
        data={
            "macro_class": "Financeiro",
            "text_input": "Preciso da segunda via do boleto com urgencia.",
        },
    )

    assert response.status_code == 200
    assert b"Nenhuma chave foi configurada para Groq" in response.data


def test_genai_demo_route_falls_back_to_mock_when_remote_provider_fails(
    monkeypatch,
    tmp_path: Path,
):
    """Garante que a interface continua operante com fallback para mock."""

    def fake_generate(self: OpenAICompatibleProvider, prompt: str) -> str:
        return self._fallback_to_mock(prompt, reason="falha de rede na chamada do provider")

    monkeypatch.setattr(
        OpenAICompatibleProvider,
        "generate_structured_completion",
        fake_generate,
    )

    app = create_app("testing")
    app.config["ARTIFACTS_FOLDER"] = tmp_path / "artifacts"
    app.config["GENAI_PROVIDER"] = "groq"
    app.config["GENAI_MOCK_MODE"] = False
    app.config["GENAI_API_KEY"] = ""
    app.config["GROQ_API_KEY"] = "groq-secret"
    app.config["GENAI_MODEL"] = "llama-3.3-70b-versatile"
    client = app.test_client()

    response = client.post(
        "/genai-demo",
        data={
            "macro_class": "Financeiro",
            "text_input": "Preciso da segunda via do boleto com urgencia.",
        },
    )

    assert response.status_code == 200
    assert b"Fallback aplicado" in response.data
    assert b"Provider efetivo" in response.data
    assert b"mock" in response.data


def test_predict_route_guides_user_when_artifacts_are_missing(tmp_path: Path):
    """Garante orientacao amigavel quando os artefatos ainda nao foram gerados."""
    app = create_app("testing")
    app.config["ARTIFACTS_FOLDER"] = tmp_path / "artifacts"
    client = app.test_client()

    response = client.get("/predict")

    assert response.status_code == 200
    assert b"Artefatos ainda nao disponiveis" in response.data
    assert b"Ir para /baseline" in response.data


def test_predict_route_runs_end_to_end_in_mock_mode(tmp_path: Path):
    """Garante a inferencia ponta a ponta com baseline treinado e mock ativo."""
    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = tmp_path
    app.config["ARTIFACTS_FOLDER"] = tmp_path / "artifacts"
    app.config["GENAI_PROVIDER"] = "mock"
    app.config["GENAI_MOCK_MODE"] = True
    client = app.test_client()

    csv_content = (
        "id_registro,texto,canal_origem,data,classe_macro,classe_detalhada\n"
        "1,Erro no login do sistema,email,2026-04-04,Suporte,Login\n"
        "2,Problema com senha esquecida,chat,2026-04-05,Suporte,Senha\n"
        "3,Duvida sobre segunda via de boleto,email,2026-04-06,Financeiro,Boleto\n"
        "4,Quero atualizar dados cadastrais,portal,2026-04-07,Cadastro,Atualizacao\n"
        "5,Erro ao anexar documento,chat,2026-04-08,Suporte,Anexo\n"
        "6,Consulta sobre cobranca em aberto,email,2026-04-09,Financeiro,Cobranca\n"
        "7,Pedido de alteracao de endereco,portal,2026-04-10,Cadastro,Endereco\n"
        "8,Problema no acesso a conta,chat,2026-04-11,Suporte,Login\n"
    )
    (tmp_path / "baseline.csv").write_text(csv_content, encoding="utf-8")

    baseline_response = client.get("/baseline")

    assert baseline_response.status_code == 200

    response = client.post(
        "/predict",
        data={
            "channel_origin": "Email",
            "text_input": "Preciso da segunda via do boleto com urgencia hoje.",
        },
    )

    assert response.status_code == 200
    assert b"Predicao estatistica" in response.data
    assert b"Classe detalhada sugerida" in response.data
    assert b"Financeiro" in response.data
    assert b"Boleto" in response.data
    assert b"Provider usado" in response.data
