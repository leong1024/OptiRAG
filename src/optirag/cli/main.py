from __future__ import annotations

import typer

from optirag.cli import data_cmd, eval_cmd, index_cmd, prompt_opt_cmd, tune_cmd

app = typer.Typer(no_args_is_help=True, name="optirag")
app.add_typer(data_cmd.app, name="data")
app.add_typer(index_cmd.app, name="index")
app.add_typer(eval_cmd.app, name="eval")
app.add_typer(tune_cmd.app, name="tune")
app.add_typer(prompt_opt_cmd.app, name="prompt-opt")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
