from pathlib import Path
import re

from pypdf import PdfReader

from app.core.constants import MAX_PDF_CHUNKS_PER_FILE, MAX_TEXT_CHUNKS_PER_FILE
from app.indexing.chunker import chunk_text
from app.models.schemas import IndexRecord
from app.embeddings.gemini import ImageDescription
from app.utils.files import build_chunk_id, build_document_id, get_image_mime_type
from app.utils.images import create_quicklook_thumbnail, create_thumbnail


def build_text_records(path: Path, thumbs_dir: Path | None = None, input_root: Path | None = None) -> list[IndexRecord]:
    resolved_path = path.resolve()
    contents = path.read_text(encoding="utf-8", errors="ignore")
    file_id = build_document_id(path)
    thumbnail_path = create_quicklook_thumbnail(path, thumbs_dir) if thumbs_dir else None
    records: list[IndexRecord] = []
    chunks = chunk_text(contents)[:MAX_TEXT_CHUNKS_PER_FILE]

    for chunk_index, chunk in enumerate(chunks):
        metadata = _base_metadata(path, file_id=file_id, modality="text", source_type="text", input_root=input_root)
        metadata["chunk_index"] = chunk_index
        metadata["preview_text"] = _preview_text(chunk)
        metadata["truncated_for_indexing"] = len(chunks) == MAX_TEXT_CHUNKS_PER_FILE
        if thumbnail_path:
            metadata["thumbnail_path"] = str(thumbnail_path)
        records.append(
                IndexRecord(
                    id=build_chunk_id(path, chunk_index),
                    file_id=file_id,
                    path=str(resolved_path),
                    modality="text",
                    document=chunk,
                    metadata=metadata,
                )
        )

    return records


def build_pdf_records(path: Path, thumbs_dir: Path | None = None, input_root: Path | None = None) -> list[IndexRecord]:
    resolved_path = path.resolve()
    pages = extract_pdf_pages(path)
    file_id = build_document_id(path)
    thumbnail_path = create_quicklook_thumbnail(path, thumbs_dir) if thumbs_dir else None
    records: list[IndexRecord] = []
    chunk_index = 0

    for page_number, page_text in pages:
        if not page_text:
            continue

        for chunk in chunk_text(page_text):
            if chunk_index >= MAX_PDF_CHUNKS_PER_FILE:
                return records
            metadata = _base_metadata(path, file_id=file_id, modality="pdf", source_type="pdf", input_root=input_root)
            metadata["chunk_index"] = chunk_index
            metadata["page_number"] = page_number
            metadata["preview_text"] = _preview_text(chunk)
            metadata["truncated_for_indexing"] = chunk_index + 1 >= MAX_PDF_CHUNKS_PER_FILE
            if thumbnail_path:
                metadata["thumbnail_path"] = str(thumbnail_path)
            records.append(
                IndexRecord(
                    id=build_chunk_id(path, chunk_index),
                    file_id=file_id,
                    path=str(resolved_path),
                    modality="pdf",
                    document=chunk,
                    metadata=metadata,
                )
            )
            chunk_index += 1

    return records


def build_pdf_direct_record(path: Path, thumbs_dir: Path | None = None, input_root: Path | None = None) -> IndexRecord:
    resolved_path = path.resolve()
    pages = extract_pdf_pages(path)
    preview_source = " ".join(text for _, text in pages if text).strip()
    file_id = build_document_id(path)
    thumbnail_path = create_quicklook_thumbnail(path, thumbs_dir) if thumbs_dir else None
    metadata = _base_metadata(path, file_id=file_id, modality="pdf", source_type="pdf", input_root=input_root)
    metadata["page_count"] = len(pages)
    metadata["embedding_mode"] = "native_pdf"
    metadata["preview_text"] = _preview_text(preview_source or path.stem)
    if thumbnail_path:
        metadata["thumbnail_path"] = str(thumbnail_path)

    return IndexRecord(
        id=file_id,
        file_id=file_id,
        path=str(resolved_path),
        modality="pdf",
        document=metadata["preview_text"],
        metadata=metadata,
    )


def build_image_record(path: Path, thumbs_dir: Path, input_root: Path | None = None) -> IndexRecord:
    resolved_path = path.resolve()
    file_id = build_document_id(path)
    thumbnail_path = create_thumbnail(path, thumbs_dir)
    metadata = _base_metadata(path, file_id=file_id, modality="image", source_type="image", input_root=input_root)
    metadata["thumbnail_path"] = str(thumbnail_path)
    metadata["mime_type"] = get_image_mime_type(path)
    metadata["preview_text"] = path.stem.replace("-", " ").replace("_", " ").strip() or path.name

    return IndexRecord(
        id=file_id,
        file_id=file_id,
        path=str(resolved_path),
        modality="image",
        document=None,
        metadata=metadata,
    )


def build_image_records(
    path: Path,
    thumbs_dir: Path,
    *,
    description: ImageDescription,
    ocr_text: str = "",
    input_root: Path | None = None,
) -> list[IndexRecord]:
    base_record = build_image_record(path, thumbs_dir, input_root=input_root)
    base_record.metadata["preview_text"] = description.caption
    base_record.metadata["image_caption"] = description.caption
    base_record.metadata["image_tags"] = ", ".join(description.tags)
    base_record.metadata["image_concepts"] = ", ".join(description.concepts)
    if ocr_text:
        base_record.metadata["ocr_text"] = ocr_text

    records: list[IndexRecord] = []
    records.append(
        _copy_image_record(
            base_record,
            record_id=base_record.file_id,
            document=description.to_index_text(),
            embedding_kind="description",
        )
    )
    if description.tags:
        records.append(
            _copy_image_record(
                base_record,
                record_id=f"{base_record.file_id}:image:tags",
                document=f"Tags: {', '.join(description.tags)}",
                embedding_kind="tags",
            )
        )
    if description.concepts:
        records.append(
            _copy_image_record(
                base_record,
                record_id=f"{base_record.file_id}:image:concepts",
                document=f"Concepts: {', '.join(description.concepts)}",
                embedding_kind="concepts",
            )
        )
    if ocr_text:
        records.append(
            _copy_image_record(
                base_record,
                record_id=f"{base_record.file_id}:image:ocr",
                document=f"OCR: {ocr_text}",
                embedding_kind="ocr",
            )
        )

    return records


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append((page_number, (page.extract_text() or "").strip()))
    return pages


def _base_metadata(
    path: Path,
    *,
    file_id: str,
    modality: str,
    source_type: str,
    input_root: Path | None = None,
) -> dict[str, str | int | float | bool]:
    resolved_path = path.resolve()
    stats = resolved_path.stat()
    metadata: dict[str, str | int | float | bool] = {
        "file_id": file_id,
        "path": str(resolved_path),
        "filename": resolved_path.name,
        "modality": modality,
        "source_type": source_type,
        "file_size": int(stats.st_size),
        "mtime": float(stats.st_mtime),
    }
    folder_path, folder_context = _folder_metadata(resolved_path, input_root)
    if folder_path:
        metadata["folder_path"] = folder_path
    if folder_context:
        metadata["folder_context"] = folder_context
    return metadata


def _preview_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit].strip()


def _copy_image_record(base_record: IndexRecord, *, record_id: str, document: str, embedding_kind: str) -> IndexRecord:
    metadata = dict(base_record.metadata)
    metadata["embedding_kind"] = embedding_kind
    return IndexRecord(
        id=record_id,
        file_id=base_record.file_id,
        path=base_record.path,
        modality="image",
        document=document,
        metadata=metadata,
    )


def _folder_metadata(path: Path, input_root: Path | None) -> tuple[str, str]:
    if input_root is None:
        return "", ""

    try:
        relative_parent = path.resolve().parent.relative_to(input_root.resolve())
    except ValueError:
        return "", ""

    if str(relative_parent) == ".":
        return "", ""

    folder_path = relative_parent.as_posix()
    context_parts: list[str] = []
    for part in relative_parent.parts:
        tokens = re.findall(r"[a-z0-9]+", part.lower().replace("-", " ").replace("_", " "))
        if tokens:
            context_parts.append(" ".join(tokens))
    return folder_path, " ".join(context_parts).strip()
