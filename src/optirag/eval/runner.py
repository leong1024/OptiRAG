from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from optirag.config.experiment import ExperimentConfig
from optirag.config.settings import get_settings
from optirag.eval.metrics import ScalarWeights, composite_scalar


@dataclass
class EvalReport:
    mean_scalar: float
    per_row_composite: list[float]
    ragas_scores: dict[str, float]
    details: dict[str, Any]


def _build_langchain_llm_embed() -> tuple[Any, Any]:
    s = get_settings()
    if not s.gemini_api_key:
        msg = "GEMINI_API_KEY required for RAGAS"
        raise ValueError(msg)
    # Model name without google_genai: prefix for LangChain
    chat_model = s.chat_model.replace("google_genai:", "").replace("google_genai/", "")
    llm = ChatGoogleGenerativeAI(
        model=chat_model,
        google_api_key=s.gemini_api_key,
        temperature=0.0,
    )
    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=s.gemini_api_key,
    )
    return llm, emb


def run_rag_eval(
    ds: Dataset,
    *,
    exp: ExperimentConfig | None = None,
    weights: ScalarWeights | None = None,
) -> EvalReport:
    """
    Single entry point for RAGAS evaluation (CLI, Optuna objective, Stage 2 tools).
    Expects columns: user_input, retrieved_contexts, response, reference.
    """
    from ragas import EvaluationDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    exp = exp or ExperimentConfig()
    w = weights or exp.ragas.weights
    llm, embeddings = _build_langchain_llm_embed()
    base_metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    name_set = set(exp.ragas.metric_names)
    metrics = [m for m in base_metrics if m.name in name_set]
    if not metrics:
        metrics = base_metrics

    eval_ds = EvaluationDataset.from_list(ds.to_list())
    result = ragas_evaluate(
        eval_ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    per_row: list[dict[str, Any]] = df.to_dict(orient="records")
    composites: list[float] = []
    for row in per_row:
        scores: dict[str, float] = {}
        for k in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
            if k in row and row[k] == row[k]:
                scores[k] = float(row[k])
        composites.append(composite_scalar(scores, weights=w))
    mean_scalar = float(sum(composites) / len(composites)) if composites else 0.0
    mk = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
    ragas_scores = {
        c: float(df[c].mean()) for c in mk if c in df.columns
    }
    return EvalReport(
        mean_scalar=mean_scalar,
        per_row_composite=composites,
        ragas_scores=ragas_scores,
        details={"num_rows": len(per_row)},
    )


def write_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
