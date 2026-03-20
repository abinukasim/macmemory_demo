import logging
import time
from collections.abc import Callable
from pathlib import Path

from app.core.config import Settings
from app.embeddings.gemini import GeminiEmbedder
from app.indexing.loaders import build_image_records, build_pdf_direct_record, build_pdf_records, build_text_records, extract_pdf_pages
from app.models.schemas import IndexedAsset, IndexRecord, IndexingSummary
from app.storage.chroma_store import ChromaStore
from app.utils.files import detect_modality, iter_supported_files
from app.utils.images import clear_directory
from app.utils.ocr import extract_text_from_image

logger = logging.getLogger(__name__)


class FolderIndexer:
    def __init__(
        self,
        settings: Settings,
        input_dir: Path | None = None,
        store: ChromaStore | None = None,
        embedder: GeminiEmbedder | None = None,
        ocr_extractor: Callable[[Path], str] | None = None,
    ) -> None:
        self.settings = settings
        self.input_dir = input_dir or settings.input_dir
        self.store = store or ChromaStore(settings)
        self.embedder = embedder or GeminiEmbedder(settings)
        self.ocr_extractor = ocr_extractor or extract_text_from_image

    def scan(self) -> IndexingSummary:
        scanned = 0
        supported = 0

        if self.input_dir.exists():
            for path in self.input_dir.rglob("*"):
                if path.is_file():
                    scanned += 1
                    if detect_modality(path):
                        supported += 1

        return IndexingSummary(
            input_dir=str(self.input_dir),
            scanned=scanned,
            supported=supported,
            skipped=scanned - supported,
        )

    def index(self, rebuild: bool = True) -> IndexingSummary:
        started_at = time.perf_counter()
        self.input_dir.mkdir(parents=True, exist_ok=True)
        summary = self.scan()

        if rebuild:
            self.store.reset()
            clear_directory(self.settings.thumbs_dir)

        indexed = 0
        text_chunks = 0
        pdf_chunks = 0
        image_files = 0
        failed = 0

        for path in iter_supported_files(self.input_dir):
            try:
                modality = detect_modality(path)
                if modality == "text":
                    assets = self._embed_records(
                        build_text_records(path, self.settings.thumbs_dir, input_root=self.input_dir),
                        title=path.name,
                    )
                    text_chunks += len(assets)
                elif modality == "pdf":
                    assets = self._embed_pdf(path)
                    pdf_chunks += len(assets)
                elif modality == "image":
                    description = self.embedder.describe_image(path)
                    ocr_text = self.ocr_extractor(path)
                    records = build_image_records(
                        path,
                        self.settings.thumbs_dir,
                        description=description,
                        ocr_text=ocr_text,
                        input_root=self.input_dir,
                    )
                    assets = self._embed_records(records, title=path.name)
                    image_files += 1
                else:
                    continue

                self.store.upsert(assets)
                indexed += len(assets)
            except Exception:
                failed += 1
                logger.exception("Failed to index %s", path)

        duration = time.perf_counter() - started_at
        logger.info(
            "Index completed: scanned=%s supported=%s indexed=%s failed=%s duration=%.2fs",
            summary.scanned,
            summary.supported,
            indexed,
            failed,
            duration,
        )

        return IndexingSummary(
            input_dir=summary.input_dir,
            scanned=summary.scanned,
            supported=summary.supported,
            skipped=summary.skipped,
            indexed=indexed,
            text_chunks=text_chunks,
            pdf_chunks=pdf_chunks,
            image_files=image_files,
            failed=failed,
            duration_seconds=duration,
        )

    def _embed_records(self, records: list[IndexRecord], title: str) -> list[IndexedAsset]:
        if not records:
            return []

        embeddings = self.embedder.embed_documents(
            [record.document or "" for record in records],
            title=title,
        )

        return [
            IndexedAsset(
                **record.model_dump(),
                embedding=embedding,
            )
            for record, embedding in zip(records, embeddings, strict=True)
        ]

    def _embed_pdf(self, path: Path) -> list[IndexedAsset]:
        page_count = len(extract_pdf_pages(path))
        chunk_assets = self._embed_records(
            build_pdf_records(path, self.settings.thumbs_dir, input_root=self.input_dir),
            title=path.name,
        )

        if 0 < page_count <= 6:
            record = build_pdf_direct_record(path, self.settings.thumbs_dir, input_root=self.input_dir)
            try:
                return [
                    IndexedAsset(
                        **record.model_dump(),
                        embedding=self.embedder.embed_pdf(path),
                    )
                ] + chunk_assets
            except RuntimeError:
                logger.warning("Falling back to text chunking for PDF %s", path.name, exc_info=True)

        return chunk_assets
