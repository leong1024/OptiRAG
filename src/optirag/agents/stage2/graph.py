"""
Stage-2 orchestrator: when `optirag[agents]` is installed, extend this module with
`create_deep_agent` (orchestrator + critique + optimize subagents) and tools in `tools.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from optirag.config.settings import get_settings

logger = logging.getLogger(__name__)


def run_stage2_prompt_loop(
    experiment_yaml: Path | None = None,
    max_rounds: int = 3,
) -> None:
    """Entry point for `optirag prompt-opt run`."""
    s = get_settings()
    _ = (experiment_yaml, max_rounds, s)
    try:
        import deepagents  # noqa: F401, PLC0415
    except ImportError:
        logger.warning(
            "deepagents not installed; Stage-2 agent loop is not run. "
            "pip install 'optirag[agents]' and implement `create_deep_agent` in this file."
        )
        return
    logger.info(
        "Deep Agents package is available. Implement orchestrator + critique + optimize "
        "using `optirag.agents.stage2.tools` and RAGAS via `run_rag_eval`."
    )
