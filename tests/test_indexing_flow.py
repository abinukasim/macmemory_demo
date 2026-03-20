from app.indexing.indexer import FolderIndexer
from app.models.schemas import SearchRequest
from app.services.search import SearchService
from app.storage.chroma_store import ChromaStore
from tests.helpers import FakeEmbedder, make_settings, write_simple_pdf, write_test_image


def test_indexing_and_search_flow_with_fake_embedder(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.input_dir.mkdir(parents=True, exist_ok=True)

    (settings.input_dir / "notes.txt").write_text(
        "Semantic retrieval helps memory systems understand meaning instead of exact keywords.",
        encoding="utf-8",
    )
    (settings.input_dir / "budget.md").write_text(
        "Budget and finance planning notes for next quarter.",
        encoding="utf-8",
    )
    write_simple_pdf(
        settings.input_dir / "paper.pdf",
        "PDF semantic retrieval research and memory systems.",
    )
    image_dir = settings.input_dir / "travel" / "sunsets"
    image_dir.mkdir(parents=True, exist_ok=True)
    write_test_image(image_dir / "sunset-beach.png")

    store = ChromaStore(settings)
    embedder = FakeEmbedder()
    summary = FolderIndexer(
        settings=settings,
        store=store,
        embedder=embedder,
        ocr_extractor=lambda _path: "Sunset beach sign with opening hours",
    ).index(rebuild=True)

    assert summary.failed == 0
    assert summary.text_chunks >= 1
    assert summary.pdf_chunks >= 1
    assert summary.image_files == 1
    assert summary.indexed >= 3

    service = SearchService(settings, store=store, embedder=embedder)

    text_hits = service.search(SearchRequest(query="semantic retrieval"))
    assert any(hit.path.endswith("notes.txt") for hit in text_hits.hits)

    pdf_hits = service.search(SearchRequest(query="pdf semantic retrieval"))
    assert any(hit.path.endswith("paper.pdf") for hit in pdf_hits.hits)

    image_hits = service.search(SearchRequest(query="sunset beach"))
    assert any(hit.path.endswith("sunset-beach.png") for hit in image_hits.hits)

    rows = store.collection.get(include=["documents", "metadatas"])
    image_rows = [
        (document, metadata)
        for document, metadata in zip(rows.get("documents") or [], rows.get("metadatas") or [])
        if metadata.get("modality") == "image"
    ]
    assert len(image_rows) >= 3

    image_metadata = image_rows[0][1]
    assert image_metadata.get("image_caption") == "sunset beach"
    assert image_metadata.get("image_tags") == "sunset, beach"
    assert image_metadata.get("ocr_text") == "Sunset beach sign with opening hours"
    assert image_metadata.get("folder_path") == "travel/sunsets"
    assert image_metadata.get("folder_context") == "travel sunsets"
    assert any(document and "Caption:" in document for document, _ in image_rows)
    assert any(document and "Tags:" in document for document, _ in image_rows)
    assert any(document and "OCR:" in document for document, _ in image_rows)
