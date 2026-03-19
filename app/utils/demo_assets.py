from pathlib import Path

from PIL import Image, ImageDraw


def seed_demo_corpus(target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)

    created = [
        _write_text_file(
            target_dir / "semantic-memory-notes.md",
            "# Semantic Memory Notes\n\n"
            "MacMemory retrieves files by meaning instead of exact keywords. "
            "Semantic retrieval helps memory systems find related notes, diagrams, and research references.\n\n"
            "Use cases include idea recall, document discovery, concept search, and cross-modal retrieval across notes and images.",
        ),
        _write_text_file(
            target_dir / "product-research-notes.md",
            "# Product Research Notes\n\n"
            "User research interviews showed that people want Spotlight-like search that understands intent. "
            "Researchers repeatedly asked for semantic search across screenshots, PDFs, and notes.\n\n"
            "Interview themes: faster recall, less folder hunting, better document discovery.",
        ),
        _write_text_file(
            target_dir / "budget-planning.txt",
            "Quarterly budget and finance planning for a small product team. "
            "Track software spend, hardware reimbursements, recruiting travel, and contractor invoices.",
        ),
        _write_text_file(
            target_dir / "trip-checklist.md",
            "# Hiking Checklist\n\nTrail snacks, water filter, weather shell, first aid, route map, headlamp, and trekking poles.",
        ),
        _write_pdf(
            target_dir / "retrieval-research.pdf",
            [
                "Research summary on semantic retrieval, multimodal embeddings, and cross-modal search systems.",
                "Embedding models map text, images, and documents into a shared vector space for better recall.",
                "Evaluation notes compare query relevance, latency, ranking quality, and retrieval consistency.",
            ],
        ),
        _write_pdf(
            target_dir / "local-search-architecture.pdf",
            [
                "Architecture overview for a local semantic search product on macOS.",
                "Indexing pipeline scans files, extracts metadata, and stores embeddings in Chroma.",
                "A local API receives natural-language queries and returns ranked semantic matches.",
                "The Raycast command presents search results with previews, thumbnails, and quick actions.",
                "Document ingestion covers markdown notes, research PDFs, screenshots, and reference images.",
                "Retrieval quality improves when text documents are chunked and PDFs preserve page-level meaning.",
                "The user experience should feel like Spotlight but operate on semantic similarity instead of keywords.",
                "Future work could include incremental sync, OCR, richer previews, and download-on-demand corpora.",
            ],
        ),
        _write_demo_image(
            target_dir / "sunset-beach.png",
            label="Sunset beach reference",
            palette=((255, 130, 70), (255, 206, 120), (22, 70, 108)),
        ),
        _write_demo_image(
            target_dir / "mountain-lake.webp",
            label="Mountain lake landscape",
            palette=((96, 148, 190), (184, 222, 255), (55, 95, 85)),
            format_name="WEBP",
        ),
        _write_demo_image(
            target_dir / "wireframe-dashboard.png",
            label="Dashboard wireframe",
            palette=((232, 236, 240), (120, 137, 164), (42, 60, 83)),
        ),
    ]

    return created


def _write_text_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_pdf(path: Path, pages: list[str]) -> Path:
    font_object_number = 3 + len(pages) * 2
    objects: list[bytes] = []
    page_refs = " ".join(f"{4 + page_index * 2} 0 R" for page_index in range(len(pages)))

    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{page_refs}] /Count {len(pages)} >>\nendobj\n".encode("utf-8"))

    for page_index, text in enumerate(pages):
        page_object_number = 4 + page_index * 2
        content_object_number = page_object_number + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT\n/F1 14 Tf\n72 720 Td\n({escaped}) Tj\nET".encode("utf-8")

        objects.append(
            (
                f"{page_object_number} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>\nendobj\n"
            ).encode("utf-8")
        )
        objects.append(
            f"{content_object_number} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("utf-8")
            + stream
            + b"\nendstream\nendobj\n"
        )

    objects.append(
        f"{font_object_number} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("utf-8")
    )

    body = bytearray(b"%PDF-1.4\n")
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
    return path


def _write_demo_image(
    path: Path,
    *,
    label: str,
    palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    format_name: str = "PNG",
) -> Path:
    sky, highlight, ground = palette
    image = Image.new("RGB", (960, 640), color=sky)
    draw = ImageDraw.Draw(image)

    for y in range(300):
        blend = tuple(min(255, int((sky[channel] * (300 - y) + highlight[channel] * y) / 300)) for channel in range(3))
        draw.line((0, y, 960, y), fill=blend)

    draw.rectangle((0, 320, 960, 640), fill=ground)
    draw.ellipse((360, 130, 600, 370), fill=highlight)
    draw.rounded_rectangle((120, 420, 840, 560), radius=28, fill=(255, 255, 255))
    draw.text((160, 470), label, fill=(20, 20, 20))
    image.save(path, format=format_name)
    return path
