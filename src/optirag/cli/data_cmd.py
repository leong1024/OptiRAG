from __future__ import annotations

from pathlib import Path

import typer

from optirag.config.settings import get_settings
from optirag.data.beir_fiqa import download_fiqa_if_needed

app = typer.Typer(no_args_is_help=True, help="Download / cache BEIR FiQA")


@app.command("download")
def download(
    data_dir: Path = typer.Option(None, help="Override OPTIRAG_DATA_DIR"),
) -> None:
    s = get_settings()
    root = data_dir or s.data_dir
    p = download_fiqa_if_needed(root)
    typer.echo(f"FiQA data at: {p}")
