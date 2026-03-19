import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from app.utils.demo_assets import _write_pdf

USER_AGENT = "MacMemoryDemo/0.1 (+local semantic search demo)"


def download_open_corpus(target_dir: Path, count: int = 100) -> dict[str, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = target_dir / "_sources"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    text_target = max(12, math.floor(count * 0.2))
    pdf_target = max(12, math.floor(count * 0.2))
    image_target = max(30, count - text_target - pdf_target)

    texts = _download_gutenberg_texts(target_dir, text_target)
    images = _download_wikimedia_media(target_dir, image_target, mime_prefix="image/")
    pdfs = _build_public_domain_pdfs(target_dir, texts[:pdf_target], pdf_target)

    manifest = {
        "texts": texts,
        "images": images,
        "pdfs": pdfs,
    }
    (metadata_dir / "open_corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "texts": len(texts),
        "images": len(images),
        "pdfs": len(pdfs),
        "total": len(texts) + len(images) + len(pdfs),
    }


def _download_gutenberg_texts(target_dir: Path, target_count: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    page = 1

    while len(results) < target_count:
        try:
            payload = _get_json(f"https://gutendex.com/books?page={page}")
        except Exception:
            break
        books = payload.get("results", [])
        if not books:
            break

        for book in books:
            formats = book.get("formats", {})
            text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")
            if not text_url:
                continue

            title = book.get("title", "untitled")
            filename = _slugify(title)[:80] or f"gutenberg-{book.get('id', len(results))}"
            destination = target_dir / f"{filename}.txt"
            if destination.exists():
                continue

            content = _download_text(text_url)
            if len(content.strip()) < 2000:
                continue

            destination.write_text(content, encoding="utf-8")
            results.append(
                {
                    "path": str(destination),
                    "title": title,
                    "source_url": text_url,
                    "license": "Project Gutenberg public domain text",
                }
            )
            if len(results) >= target_count:
                break

        page += 1

    return results


def _download_wikimedia_media(target_dir: Path, target_count: int, mime_prefix: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    while len(results) < target_count:
        query = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "generator": "random",
                "grnnamespace": "6",
                "grnlimit": "50",
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
            }
        )
        payload = _get_json(f"https://commons.wikimedia.org/w/api.php?{query}")
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            break

        for page in pages:
            imageinfo = (page.get("imageinfo") or [{}])[0]
            url = imageinfo.get("url")
            mime = imageinfo.get("mime", "")
            if not url or not mime.startswith(mime_prefix):
                continue
            if url in seen_urls:
                continue

            title = page.get("title", "untitled")
            extension = Path(urllib.parse.urlparse(url).path).suffix or _extension_for_mime(mime)
            destination = target_dir / f"{_slugify(title.replace('File:', ''))[:80]}{extension.lower()}"
            if destination.exists():
                continue

            try:
                binary = _download_bytes(url)
            except Exception:
                continue

            destination.write_bytes(binary)
            seen_urls.add(url)
            extmetadata = imageinfo.get("extmetadata", {})
            results.append(
                {
                    "path": str(destination),
                    "title": title,
                    "source_url": url,
                    "license": _extmetadata_value(extmetadata, "LicenseShortName") or "Wikimedia Commons free media",
                    "license_url": _extmetadata_value(extmetadata, "LicenseUrl") or "",
                }
            )
            if len(results) >= target_count:
                break

    return results


def _build_public_domain_pdfs(target_dir: Path, text_sources: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for index, source in enumerate(text_sources[:target_count], start=1):
        text = Path(source["path"]).read_text(encoding="utf-8", errors="ignore")
        chunks = _paragraph_chunks(text, pages=8)
        if not chunks:
            continue

        destination = target_dir / f"{Path(source['path']).stem}.pdf"
        _write_pdf(destination, chunks)
        results.append(
            {
                "path": str(destination),
                "title": f"{source['title']} (PDF excerpt)",
                "source_url": source["source_url"],
                "license": source["license"],
            }
        )

    return results


def _paragraph_chunks(text: str, pages: int) -> list[str]:
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current} {paragraph}".strip()
        if len(candidate) > 500 and current:
            chunks.append(current)
            current = paragraph
            if len(chunks) >= pages:
                break
        else:
            current = candidate

    if current and len(chunks) < pages:
        chunks.append(current)

    return chunks[:pages]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _download_text(url: str) -> str:
    return _with_retries(lambda: _fetch(url, timeout=45).decode("utf-8", errors="ignore"))


def _download_bytes(url: str) -> bytes:
    return _with_retries(lambda: _fetch(url, timeout=60))


def _get_json(url: str) -> dict:
    return _with_retries(lambda: json.loads(_fetch(url, timeout=30).decode("utf-8")))


def _fetch(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _with_retries(func, attempts: int = 3, delay_seconds: float = 1.0):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error


def _extension_for_mime(mime: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime, ".bin")


def _extmetadata_value(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return None
