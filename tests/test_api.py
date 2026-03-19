from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.models.schemas import HealthResponse
from app.services.search import SearchService
from tests.helpers import FakeEmbedder, FakeStore, make_settings


def test_health_returns_configured_paths(tmp_path) -> None:
    settings = make_settings(tmp_path)
    app.dependency_overrides[routes.get_settings] = lambda: settings

    client = TestClient(app)
    response = client.get("/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = HealthResponse(**response.json())
    assert payload.collection == settings.chroma_collection
    assert payload.input_dir == str(settings.input_dir)


def test_search_rejects_whitespace_only_queries() -> None:
    client = TestClient(app)

    response = client.post("/search", json={"query": "   ", "k": 5})

    assert response.status_code == 422


def test_search_returns_empty_hits_when_not_indexed(tmp_path) -> None:
    settings = make_settings(tmp_path)
    service = SearchService(settings, store=FakeStore([]), embedder=FakeEmbedder())
    app.dependency_overrides[routes.get_search_service] = lambda: service

    client = TestClient(app)
    response = client.post("/search", json={"query": "semantic retrieval", "k": 5})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["hits"] == []
