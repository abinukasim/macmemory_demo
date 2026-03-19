from pathlib import Path

import typer

from app.core.config import get_settings
from app.indexing.indexer import FolderIndexer

cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    input_dir: Path | None = typer.Option(default=None, help="Override the configured input directory."),
    rebuild: bool = typer.Option(True, "--rebuild/--no-rebuild", help="Rebuild the local index from scratch."),
) -> None:
    """Index the configured input folder into the local vector store."""
    settings = get_settings()
    summary = FolderIndexer(settings=settings, input_dir=input_dir or settings.input_dir).index(rebuild=rebuild)

    typer.echo(f"Scanned:   {summary.scanned}")
    typer.echo(f"Supported: {summary.supported}")
    typer.echo(f"Skipped:   {summary.skipped}")
    typer.echo(f"Indexed:   {summary.indexed}")
    typer.echo(f"Text:      {summary.text_chunks}")
    typer.echo(f"PDF:       {summary.pdf_chunks}")
    typer.echo(f"Images:    {summary.image_files}")
    typer.echo(f"Failed:    {summary.failed}")
    typer.echo(f"Duration:  {summary.duration_seconds:.2f}s")


if __name__ == "__main__":
    cli()
