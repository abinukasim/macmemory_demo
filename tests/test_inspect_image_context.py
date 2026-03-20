from app.cli.inspect_image_context import _find_image_records
from app.models.schemas import IndexedAsset
from app.storage.chroma_store import ChromaStore
from tests.helpers import make_settings


def test_find_image_records_returns_only_matching_image_filename(tmp_path) -> None:
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.reset()
    store.upsert(
        [
            IndexedAsset(
                id="image-a",
                file_id="image-a",
                path="/tmp/christmas_tree.jpeg",
                modality="image",
                document="Caption: christmas tree",
                metadata={
                    "file_id": "image-a",
                    "path": "/tmp/christmas_tree.jpeg",
                    "filename": "christmas_tree.jpeg",
                    "modality": "image",
                    "image_caption": "christmas tree",
                    "embedding_kind": "description",
                },
                embedding=[1.0, 0.0],
            ),
            IndexedAsset(
                id="pdf-a",
                file_id="pdf-a",
                path="/tmp/notes.pdf",
                modality="pdf",
                document="holiday planning notes",
                metadata={
                    "file_id": "pdf-a",
                    "path": "/tmp/notes.pdf",
                    "filename": "notes.pdf",
                    "modality": "pdf",
                },
                embedding=[0.0, 1.0],
            ),
        ]
    )

    matches = _find_image_records(store, "christmas_tree.jpeg")

    assert len(matches) == 1
    assert matches[0][1]["filename"] == "christmas_tree.jpeg"
    assert matches[0][1]["modality"] == "image"
