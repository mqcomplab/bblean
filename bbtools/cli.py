r"""Command line interface entrypoints"""
from typer import Typer

app = Typer()


@app.command()
def _main() -> None:
    pass
