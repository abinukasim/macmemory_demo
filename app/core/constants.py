TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
PDF_EXTENSIONS = {".pdf"}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS
SUPPORTED_MODALITIES = ("text", "image", "pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 8
MAX_TEXT_CHUNKS_PER_FILE = 64
MAX_PDF_CHUNKS_PER_FILE = 64
OCR_MIN_TEXT_CHARS = 24
OCR_MAX_TEXT_CHARS = 1200

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
}
