from pathlib import Path

import typer

from app.core.config import get_settings
from app.utils.demo_assets import seed_demo_corpus

cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    output_dir: Path | None = typer.Option(default=None, help="Override the configured input directory."),
) -> None:
    """Create a small demo corpus with text, PDF, and image files."""
    settings = get_settings()
    target_dir = output_dir or settings.input_dir
    created = seed_demo_corpus(target_dir)

    typer.echo(f"Created {len(created)} demo files in {target_dir}")
    for path in created:
        typer.echo(f"- {path.name}")


if __name__ == "__main__":
    cli()
