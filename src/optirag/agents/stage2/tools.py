"""Tools for Deep Agents (eval, prompt I/O). Wire to `eval.runner.run_rag_eval`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from optirag.config.experiment import load_experiment
from optirag.eval.runner import run_rag_eval


def tool_run_ragas_on_dataset(rows: list[dict[str, Any]], experiment_path: str | Path) -> dict[str, Any]:
    exp = load_experiment(Path(experiment_path))
    ds = Dataset.from_list(rows)
    report = run_rag_eval(ds, exp=exp)
    return {
        "mean_scalar": report.mean_scalar,
        "ragas_scores": report.ragas_scores,
    }


def tool_load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def tool_save_prompt(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
