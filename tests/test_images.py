import io
from pathlib import Path

from PIL import Image

from app.embeddings.gemini import _parse_image_description
from app.utils.images import normalize_image_for_embedding
from tests.helpers import write_test_image


def test_normalize_image_for_embedding_reencodes_png(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    write_test_image(path)

    image_bytes, mime_type = normalize_image_for_embedding(path)

    assert mime_type in {"image/png", "image/jpeg"}
    assert image_bytes


def test_normalize_image_for_embedding_resizes_large_image(tmp_path: Path) -> None:
    path = tmp_path / "large.jpg"
    Image.new("RGB", (4000, 3000), color=(50, 100, 150)).save(path, format="JPEG")

    image_bytes, mime_type = normalize_image_for_embedding(path, size=(512, 512))

    assert mime_type == "image/jpeg"
    with Image.open(io.BytesIO(image_bytes)) as image:
        assert max(image.size) <= 512


def test_normalize_image_for_embedding_converts_webp(tmp_path: Path) -> None:
    path = tmp_path / "sample.webp"
    write_test_image(path)

    image_bytes, mime_type = normalize_image_for_embedding(path)

    assert mime_type in {"image/png", "image/jpeg"}
    assert image_bytes != path.read_bytes()


def test_parse_image_description_extracts_caption_tags_and_concepts() -> None:
    description = _parse_image_description(
        "caption: cozy living room with christmas tree by fireplace\n"
        "tags: christmas, holiday, tree, fireplace, ornaments, gifts\n"
        "concepts: holiday celebration, indoor decor, family gathering\n"
    )

    assert description is not None
    assert description.caption == "cozy living room with christmas tree by fireplace"
    assert "christmas" in description.tags
    assert "holiday celebration" in description.concepts
