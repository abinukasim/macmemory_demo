from collections import defaultdict

import typer

from app.core.config import get_settings
from app.storage.chroma_store import ChromaStore

cli = typer.Typer(add_completion=False)


@cli.command()
def main(filename: str = typer.Argument(..., help="Image filename to inspect, including extension.")) -> None:
    """Inspect the stored context and indexed records for one image filename."""
    settings = get_settings()
    store = ChromaStore(settings)
    matches = _find_image_records(store, filename)

    if not matches:
        typer.echo(f"No indexed image records found for {filename}")
        raise typer.Exit(code=1)

    grouped: dict[str, list[tuple[str | None, dict[str, object]]]] = defaultdict(list)
    for document, metadata in matches:
        grouped[str(metadata.get("path", ""))].append((document, metadata))

    typer.echo(f"Found {len(matches)} indexed image record(s) for {filename}")
    for path, records in grouped.items():
        base = records[0][1]
        typer.echo("=" * 80)
        typer.echo(f"file: {base.get('filename')}")
        typer.echo(f"path: {path}")
        typer.echo(f"folder_path: {base.get('folder_path', '')}")
        typer.echo(f"folder_context: {base.get('folder_context', '')}")
        typer.echo(f"caption: {base.get('image_caption', '')}")
        typer.echo(f"tags: {base.get('image_tags', '')}")
        typer.echo(f"concepts: {base.get('image_concepts', '')}")
        typer.echo(f"ocr_text: {base.get('ocr_text', '')}")
        typer.echo(f"thumbnail_path: {base.get('thumbnail_path', '')}")
        typer.echo(f"records: {len(records)}")

        for index, (document, metadata) in enumerate(records, start=1):
            typer.echo("-" * 80)
            typer.echo(f"record {index}:")
            typer.echo(f"  id: {metadata.get('file_id', '')}")
            typer.echo(f"  embedding_kind: {metadata.get('embedding_kind', 'description')}")
            typer.echo("  document:")
            typer.echo(_indent(document or ""))


def _find_image_records(store: ChromaStore, filename: str) -> list[tuple[str | None, dict[str, object]]]:
    rows = store.collection.get(include=["documents", "metadatas"])
    documents = rows.get("documents") or []
    metadatas = rows.get("metadatas") or []
    matches: list[tuple[str | None, dict[str, object]]] = []

    for document, metadata in zip(documents, metadatas):
        if not metadata:
            continue
        if metadata.get("modality") != "image":
            continue
        if str(metadata.get("filename", "")) != filename:
            continue
        matches.append((document, dict(metadata)))

    return matches


def _indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines() or [""])


if __name__ == "__main__":
    cli()
