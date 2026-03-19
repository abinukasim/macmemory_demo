from hashlib import sha1
from pathlib import Path
from typing import Iterator, Literal

from app.core.constants import IMAGE_EXTENSIONS, IMAGE_MIME_TYPES, PDF_EXTENSIONS, SUPPORTED_EXTENSIONS, TEXT_EXTENSIONS


def detect_modality(path: Path) -> Literal["text", "image", "pdf"] | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    return None


def iter_supported_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def build_document_id(path: Path) -> str:
    return sha1(str(path.resolve()).encode("utf-8")).hexdigest()


def build_chunk_id(path: Path, chunk_index: int) -> str:
    return f"{build_document_id(path)}:chunk:{chunk_index}"


def get_image_mime_type(path: Path) -> str:
    return IMAGE_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
