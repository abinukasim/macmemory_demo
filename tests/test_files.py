from pathlib import Path

from app.utils.files import build_document_id, detect_modality


def test_detect_modality_for_supported_types() -> None:
    assert detect_modality(Path("notes.md")) == "text"
    assert detect_modality(Path("photo.png")) == "image"
    assert detect_modality(Path("paper.pdf")) == "pdf"
    assert detect_modality(Path("archive.zip")) is None


def test_build_document_id_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    assert build_document_id(path) == build_document_id(path)
