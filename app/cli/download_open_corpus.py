from pathlib import Path

import typer

from app.core.config import get_settings
from app.utils.open_corpus import download_open_corpus

cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    output_dir: Path | None = typer.Option(default=None, help="Override the configured input directory."),
    count: int = typer.Option(default=100, min=10, max=200, help="Target number of files to download."),
) -> None:
    """Download a real open-license/public-domain corpus for demo testing."""
    settings = get_settings()
    target_dir = output_dir or settings.input_dir
    summary = download_open_corpus(target_dir, count=count)

    typer.echo(f"Downloaded corpus into {target_dir}")
    typer.echo(f"Texts:  {summary['texts']}")
    typer.echo(f"Images: {summary['images']}")
    typer.echo(f"PDFs:   {summary['pdfs']}")
    typer.echo(f"Total:  {summary['total']}")


if __name__ == "__main__":
    cli()
