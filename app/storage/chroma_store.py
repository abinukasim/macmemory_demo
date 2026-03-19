from typing import Any

import chromadb

from app.core.config import Settings
from app.models.schemas import IndexedAsset, QueryMatch


class ChromaStore:
    """Thin wrapper around a single local persistent Chroma collection."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.persist_dir = settings.chroma_dir
        self.collection_name = settings.chroma_collection
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset(self) -> None:
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None
        _ = self.collection

    def count(self) -> int:
        return int(self.collection.count())

    def upsert(self, assets: list[IndexedAsset]) -> None:
        if not assets:
            return

        self.collection.upsert(
            ids=[asset.id for asset in assets],
            embeddings=[asset.embedding for asset in assets],
            documents=[asset.document for asset in assets],
            metadatas=[asset.metadata for asset in assets],
        )

    def query(self, query_embedding: list[float], limit: int) -> list[QueryMatch]:
        if self.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        matches: list[QueryMatch] = []
        for index, match_id in enumerate(ids):
            matches.append(
                QueryMatch(
                    id=match_id,
                    document=documents[index] if index < len(documents) else None,
                    metadata=metadatas[index] if index < len(metadatas) and metadatas[index] else {},
                    distance=float(distances[index]) if index < len(distances) else 0.0,
                )
            )

        return matches
