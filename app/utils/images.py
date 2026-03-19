import io
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from pillow_heif import register_heif_opener

from app.utils.files import build_document_id

register_heif_opener()


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def create_thumbnail(source_path: Path, thumbs_dir: Path, size: tuple[int, int] = (512, 512)) -> Path:
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    destination = (thumbs_dir / f"{build_document_id(source_path)}.jpg").resolve()

    with Image.open(source_path) as image:
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail(size)
        thumbnail.save(destination, format="JPEG", quality=85)

    return destination


def create_quicklook_thumbnail(source_path: Path, thumbs_dir: Path) -> Path | None:
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    destination = (thumbs_dir / f"{build_document_id(source_path)}.png").resolve()

    with TemporaryDirectory() as temp_dir:
        result = subprocess.run(
            [
                "qlmanage",
                "-t",
                "-s",
                "512",
                "-o",
                temp_dir,
                str(source_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None

        generated = list(Path(temp_dir).glob("*.png"))
        if not generated:
            return None

        shutil.move(str(generated[0]), destination)
        return destination


def normalize_image_for_embedding(source_path: Path, size: tuple[int, int] = (1600, 1600)) -> tuple[bytes, str]:
    with Image.open(source_path) as image:
        image.thumbnail(size)
        converted = image.convert("RGBA" if _should_keep_alpha(image) else "RGB")
        buffer = io.BytesIO()
        if converted.mode == "RGBA":
            converted.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue(), "image/png"

        converted.save(buffer, format="JPEG", quality=88, optimize=True)
        return buffer.getvalue(), "image/jpeg"


def _should_keep_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or "transparency" in image.info
