from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, help="Stage-2 prompt optimization")


@app.command("run")
def run_prompt_opt() -> None:
    try:
        from optirag.agents.stage2.graph import run_stage2_prompt_loop

        run_stage2_prompt_loop()
    except ImportError as e:
        typer.echo(
            "Install agents extra: pip install 'optirag[agents]' and configure keys. "
            f"({e})",
            err=True,
        )
        raise typer.Exit(code=1) from e
