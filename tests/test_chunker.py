from app.indexing.chunker import chunk_text


def test_chunk_text_uses_overlap_for_long_content() -> None:
    text = "Paragraph one. " * 120

    chunks = chunk_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert chunks[0][-20:] in chunks[1]


def test_chunk_text_handles_empty_input() -> None:
    assert chunk_text("   \n\t  ") == []
