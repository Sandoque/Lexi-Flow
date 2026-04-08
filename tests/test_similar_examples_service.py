"""Testes da recuperacao de exemplos similares para few-shot contextual."""

from pathlib import Path

from app.services.nlp_config import obter_configuracao_nlp_padrao
from app.services.similar_examples_service import recuperar_exemplos_similares


def test_recuperar_exemplos_similares_prioriza_mesma_macro(tmp_path: Path):
    """A busca deve priorizar exemplos da macro prevista quando houver dados suficientes."""
    dataset_path = tmp_path / "historico.csv"
    dataset_path.write_text(
        (
            "id_registro,texto,canal_origem,data,classe_macro,classe_detalhada\n"
            "1,Preciso da segunda via do boleto,email,2026-04-01,Financeiro,Boleto\n"
            "2,Existe cobranca indevida na minha fatura,chat,2026-04-02,Financeiro,Cobranca\n"
            "3,Nao consigo redefinir a senha,portal,2026-04-03,Suporte,Senha\n"
        ),
        encoding="utf-8",
    )

    result = recuperar_exemplos_similares(
        text="Preciso emitir um novo boleto hoje.",
        predicted_macro="Financeiro",
        dataset_path=dataset_path,
        nlp_config=obter_configuracao_nlp_padrao(),
        top_k=2,
        restrict_to_macro=True,
    )

    assert result["used_count"] == 2
    assert result["scope"] == "macro"
    assert all(example.macro_class == "Financeiro" for example in result["few_shot_examples"])


def test_recuperar_exemplos_similares_retornando_estrutura_vazia_para_texto_vazio(tmp_path: Path):
    """Texto vazio nao deve quebrar a recuperacao de similares."""
    dataset_path = tmp_path / "historico.csv"
    dataset_path.write_text(
        (
            "id_registro,texto,canal_origem,data,classe_macro,classe_detalhada\n"
            "1,Preciso da segunda via do boleto,email,2026-04-01,Financeiro,Boleto\n"
        ),
        encoding="utf-8",
    )

    result = recuperar_exemplos_similares(
        text="",
        predicted_macro="Financeiro",
        dataset_path=dataset_path,
        nlp_config=obter_configuracao_nlp_padrao(),
    )

    assert result["used_count"] == 0
    assert result["few_shot_examples"] == []


def test_recuperar_exemplos_similares_remove_duplicatas_do_historico(tmp_path: Path):
    """A seleção deve evitar repetir o mesmo caso histórico no few-shot contextual."""
    dataset_path = tmp_path / "historico.csv"
    dataset_path.write_text(
        (
            "id_registro,texto,canal_origem,data,classe_macro,classe_detalhada\n"
            "1,Efetuei o pagamento mas consta pendente,email,2026-04-01,Financeiro,Pagamento\n"
            "2,Efetuei o pagamento mas consta pendente,chat,2026-04-02,Financeiro,Pagamento\n"
            "3,Preciso da segunda via do boleto,portal,2026-04-03,Financeiro,Boleto\n"
            "4,Existe cobranca indevida na fatura,portal,2026-04-04,Financeiro,Cobranca\n"
        ),
        encoding="utf-8",
    )

    result = recuperar_exemplos_similares(
        text="Efetuei o pagamento ontem, mas segue pendente no sistema.",
        predicted_macro="Financeiro",
        dataset_path=dataset_path,
        nlp_config=obter_configuracao_nlp_padrao(),
        top_k=3,
        restrict_to_macro=True,
    )

    returned_texts = [item["text"] for item in result["support_examples"]]

    assert result["used_count"] == 3
    assert len(returned_texts) == len(set(returned_texts))
