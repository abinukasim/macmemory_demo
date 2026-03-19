from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from app.core.config import Settings
from app.embeddings.gemini import ImageDescription

KEYWORDS = (
    "semantic",
    "retrieval",
    "memory",
    "budget",
    "finance",
    "sunset",
    "beach",
    "pdf",
)


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return _keyword_vector(query)

    def embed_documents(self, contents: list[str], title: str | None = None) -> list[list[float]]:
        return [_keyword_vector(content) for content in contents]

    def embed_image(self, path: Path) -> list[float]:
        return self.embed_documents([self.describe_image(path).to_index_text()], title=path.name)[0]

    def caption_image(self, path: Path) -> str:
        return self.describe_image(path).caption

    def describe_image(self, path: Path) -> ImageDescription:
        stem = path.stem.replace("-", " ").replace("_", " ")
        tags = [token for token in stem.lower().split() if token]
        return ImageDescription(
            caption=stem,
            tags=tags[:10],
            concepts=["image search"] if tags else [],
        )

    def embed_pdf(self, path: Path) -> list[float]:
        text = " ".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
        return _keyword_vector(text)


class FakeStore:
    def __init__(self, matches: list) -> None:
        self.matches = matches

    def count(self) -> int:
        return len(self.matches)

    def query(self, query_embedding: list[float], limit: int):
        return self.matches[:limit]


def make_settings(tmp_path: Path) -> Settings:
    return Settings.model_construct(
        gemini_api_key="test-key",
        gemini_embedding_model="test-model",
        input_dir=tmp_path / "input",
        chroma_dir=tmp_path / "chroma",
        thumbs_dir=tmp_path / "thumbs",
        chroma_collection="macmemory-test",
        api_host="127.0.0.1",
        api_port=8000,
    )


def write_simple_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped}) Tj\nET".encode("utf-8")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n"
        ),
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("utf-8") + stream + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    body.extend(f"xref\n0 {len(offsets)}\n".encode("utf-8"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))

    body.extend(
        (
            f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("utf-8")
    )
    path.write_bytes(body)


def write_test_image(path: Path, color: tuple[int, int, int] = (240, 120, 80)) -> None:
    image = Image.new("RGB", (64, 64), color=color)
    format_name = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
    }.get(path.suffix.lower(), "PNG")
    image.save(path, format=format_name)


def _keyword_vector(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(keyword)) for keyword in KEYWORDS]
