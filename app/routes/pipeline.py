"""Rotas relacionadas ao fluxo de classificacao textual."""

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for

from app.services.baseline_classifier import (
    BaselineError,
    executar_treinamento_baseline,
    get_baseline_placeholder,
)
from app.services.data_validation import get_validation_placeholder
from app.services.eda_service import EDAError, carregar_eda_do_ultimo_dataset
from app.services.exploratory_analysis import get_eda_placeholder
from app.services.genai_refiner import (
    GenAIRefiner,
    GenAIRefinerError,
    get_demo_few_shot_examples,
    get_demo_macro_options,
    get_genai_settings_from_config,
)
from app.services.generative_refinement import get_generative_placeholder
from app.services.ingest_service import IngestaoError, REQUIRED_COLUMNS, processar_upload_csv
from app.services.prediction_service import (
    PredictionError,
    artefatos_predicao_disponiveis,
    executar_fluxo_predicao,
    obter_canais_origem_padrao,
)
from app.services.routing_service import obter_legenda_fluxo_operacional
from app.utils.dataset_locator import resolve_dataset_source

pipeline_bp = Blueprint("pipeline", __name__)


def obter_fonte_dataset_ativa() -> str:
    """Resolve e persiste a fonte ativa de dados para a sessao atual."""
    source = resolve_dataset_source(
        requested_source=request.values.get("dataset_source"),
        current_source=session.get("dataset_source"),
        use_demo_by_default=bool(current_app.config["USE_DEMO_DATASET_BY_DEFAULT"]),
    )
    session["dataset_source"] = source
    return source


def montar_contexto_fonte_dataset(dataset_source: str) -> dict:
    """Monta contexto simples para indicar a fonte ativa na interface."""
    return {
        "dataset_source": dataset_source,
        "dataset_source_label": "dataset demo" if dataset_source == "demo" else "ultimo upload",
        "using_demo_dataset": dataset_source == "demo",
        "demo_dataset_path": str(current_app.config["DEMO_DATASET_PATH"]),
    }


@pipeline_bp.route("/upload", methods=["GET", "POST"])
def upload():
    """Exibe a tela de upload e processa a ingestao do CSV."""
    dataset_source = obter_fonte_dataset_ativa()
    ingest_result = None

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")

        try:
            ingest_result = processar_upload_csv(
                uploaded_file=uploaded_file,
                upload_folder=current_app.config["UPLOAD_FOLDER"],
                allowed_extensions=current_app.config["ALLOWED_EXTENSIONS"],
            )
            session["last_uploaded_file"] = ingest_result["saved_path"]
            flash(
                "Arquivo validado e salvo com sucesso em data/raw.",
                "success",
            )
        except IngestaoError as exc:
            flash(str(exc), "danger")
        except Exception:
            current_app.logger.exception("Erro inesperado durante a ingestao do dataset.")
            flash(
                "Ocorreu um erro inesperado durante a ingestao. Tente novamente com outro arquivo.",
                "danger",
            )

    return render_template(
        "upload.html",
        ingest_result=ingest_result,
        expected_columns=REQUIRED_COLUMNS,
        **montar_contexto_fonte_dataset(dataset_source),
    )


@pipeline_bp.get("/eda")
def eda():
    """Exibe a analise exploratoria do ultimo dataset disponivel."""
    dataset_source = obter_fonte_dataset_ativa()
    try:
        eda_result = carregar_eda_do_ultimo_dataset(
            upload_folder=current_app.config["UPLOAD_FOLDER"],
            preferred_path=session.get("last_uploaded_file"),
            demo_dataset_path=current_app.config["DEMO_DATASET_PATH"],
            dataset_source=dataset_source,
            use_demo_by_default=bool(current_app.config["USE_DEMO_DATASET_BY_DEFAULT"]),
            example_limit=2,
            top_detailed_limit=8,
        )
    except EDAError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("pipeline.upload"))
    except Exception:
        current_app.logger.exception("Erro inesperado ao gerar a analise exploratoria.")
        flash(
            "Nao foi possivel montar a analise exploratoria do dataset.",
            "danger",
        )
        return redirect(url_for("pipeline.upload"))

    return render_template(
        "eda.html",
        eda_result=eda_result,
        **montar_contexto_fonte_dataset(dataset_source),
    )


@pipeline_bp.route("/baseline", methods=["GET"])
def baseline():
    """Treina e exibe o baseline hierarquico para o dataset ativo."""
    dataset_source = obter_fonte_dataset_ativa()

    try:
        baseline_result = executar_treinamento_baseline(
            upload_folder=current_app.config["UPLOAD_FOLDER"],
            artifacts_folder=current_app.config["ARTIFACTS_FOLDER"],
            preferred_path=session.get("last_uploaded_file"),
            demo_dataset_path=current_app.config["DEMO_DATASET_PATH"],
            dataset_source=dataset_source,
            use_demo_by_default=bool(current_app.config["USE_DEMO_DATASET_BY_DEFAULT"]),
        )
    except BaselineError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("pipeline.upload"))
    except Exception:
        current_app.logger.exception("Erro inesperado durante o treinamento baseline.")
        flash("Nao foi possivel treinar o baseline com o dataset atual.", "danger")
        return redirect(url_for("pipeline.upload"))

    return render_template(
        "baseline.html",
        baseline_result=baseline_result,
        **montar_contexto_fonte_dataset(dataset_source),
    )


@pipeline_bp.route("/genai-demo", methods=["GET", "POST"])
def genai_demo():
    """Executa uma demonstracao da camada GenAI complementar ao baseline."""
    macro_options = get_demo_macro_options(current_app.config["ARTIFACTS_FOLDER"])
    selected_macro = request.form.get("macro_class", macro_options[0]["macro_class"] if macro_options else "")
    selected_entry = next(
        (item for item in macro_options if item["macro_class"] == selected_macro),
        macro_options[0] if macro_options else {"macro_class": "", "detail_options": []},
    )
    text_input = request.form.get("text_input", "")
    genai_result = None

    if request.method == "POST":
        try:
            refiner = GenAIRefiner(get_genai_settings_from_config(current_app.config))
            genai_result = refiner.refine(
                text=text_input,
                predicted_macro=selected_macro,
                valid_detailed_classes=selected_entry["detail_options"],
                few_shot_examples=get_demo_few_shot_examples(),
            )
            flash(
                "A camada GenAI retornou uma sugestao estruturada para complementar o baseline.",
                "success",
            )
        except GenAIRefinerError as exc:
            flash(str(exc), "warning")
        except Exception:
            current_app.logger.exception("Erro inesperado durante a demonstracao GenAI.")
            flash("Nao foi possivel executar a demonstracao GenAI.", "danger")

    return render_template(
        "genai_demo.html",
        macro_options=macro_options,
        selected_macro=selected_macro,
        selected_entry=selected_entry,
        text_input=text_input,
        genai_result=genai_result,
    )


@pipeline_bp.route("/predict", methods=["GET", "POST"])
def predict():
    """Executa a inferencia ponta a ponta com baseline e camada GenAI complementar."""
    dataset_source = obter_fonte_dataset_ativa()
    channel_options = obter_canais_origem_padrao()
    routing_legend = obter_legenda_fluxo_operacional()
    text_input = request.form.get("text_input", "")
    selected_channel = request.form.get("channel_origin", "")
    prediction_result = None
    artifacts_ready = artefatos_predicao_disponiveis(current_app.config["ARTIFACTS_FOLDER"])

    if request.method == "POST":
        try:
            prediction_result = executar_fluxo_predicao(
                text=text_input,
                channel_origin=selected_channel,
                artifacts_folder=current_app.config["ARTIFACTS_FOLDER"],
                genai_settings=get_genai_settings_from_config(current_app.config),
            )
            if prediction_result["genai"]["status"] == "ok":
                flash("Inferencia concluida com baseline e camada GenAI complementar.", "success")
            else:
                flash(
                    "Inferencia concluida com baseline. A camada GenAI ficou indisponivel nesta execucao.",
                    "warning",
                )
        except PredictionError as exc:
            flash(str(exc), "warning")
        except Exception:
            current_app.logger.exception("Erro inesperado durante a inferencia ponta a ponta.")
            flash("Nao foi possivel concluir a inferencia com o texto informado.", "danger")

    return render_template(
        "predict.html",
        channel_options=channel_options,
        text_input=text_input,
        selected_channel=selected_channel,
        prediction_result=prediction_result,
        artifacts_ready=artifacts_ready,
        routing_legend=routing_legend,
        **montar_contexto_fonte_dataset(dataset_source),
    )


@pipeline_bp.get("/results")
def results():
    """Renderiza uma visao placeholder para as saidas do fluxo."""
    context = {
        "validation_summary": get_validation_placeholder(),
        "eda_summary": get_eda_placeholder(),
        "baseline_summary": get_baseline_placeholder(),
        "generative_summary": get_generative_placeholder(),
    }
    return render_template("results.html", **context)
