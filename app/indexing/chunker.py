from app.core.constants import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        max_end = min(start + chunk_size, text_length)
        end = _prefer_boundary(normalized, start, max_end)
        if end <= start:
            end = max_end

        chunk = " ".join(normalized[start:end].split()).strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def _prefer_boundary(text: str, start: int, max_end: int) -> int:
    if max_end >= len(text):
        return len(text)

    boundary_window = text[start:max_end]
    preferred_breaks = ("\n\n", "\n", ". ", "? ", "! ", "; ")

    for marker in preferred_breaks:
        index = boundary_window.rfind(marker)
        if index != -1 and start + index + len(marker) > start:
            return start + index + len(marker)

    fallback_index = boundary_window.rfind(" ")
    if fallback_index > 0:
        return start + fallback_index
    return max_end
