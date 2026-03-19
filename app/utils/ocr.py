import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

from app.core.constants import OCR_MAX_TEXT_CHARS, OCR_MIN_TEXT_CHARS


def extract_text_from_image(path: Path) -> str:
    if shutil.which("tesseract") is None:
        return ""

    with TemporaryDirectory() as temp_dir:
        prepared_path = Path(temp_dir) / "ocr-input.png"
        _prepare_image_for_ocr(path, prepared_path)

        result = subprocess.run(
            [
                "tesseract",
                str(prepared_path),
                "stdout",
                "--psm",
                "6",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return ""

    return _normalize_ocr_text(result.stdout)


def _prepare_image_for_ocr(source_path: Path, destination: Path) -> None:
    with Image.open(source_path) as image:
        grayscale = ImageOps.grayscale(image)
        contrast = ImageOps.autocontrast(grayscale)
        width, height = contrast.size
        if max(width, height) < 1800:
            scale = 1800 / max(width, height)
            contrast = contrast.resize((int(width * scale), int(height * scale)))
        contrast.save(destination, format="PNG")


def _normalize_ocr_text(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    if len(normalized) < OCR_MIN_TEXT_CHARS:
        return ""
    return normalized[:OCR_MAX_TEXT_CHARS]
