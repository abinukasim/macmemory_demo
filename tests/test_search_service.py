from app.models.schemas import QueryMatch, SearchRequest
from app.services.search import SearchService
from tests.helpers import FakeEmbedder, FakeStore, make_settings


def test_search_service_dedupes_to_file_level(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="file-a:chunk:0",
            document="semantic retrieval introduction",
            metadata={
                "file_id": "file-a",
                "path": "/tmp/file-a.txt",
                "filename": "file-a.txt",
                "modality": "text",
                "preview_text": "semantic retrieval introduction",
            },
            distance=0.1,
        ),
        QueryMatch(
            id="file-a:chunk:1",
            document="less relevant chunk",
            metadata={
                "file_id": "file-a",
                "path": "/tmp/file-a.txt",
                "filename": "file-a.txt",
                "modality": "text",
                "preview_text": "less relevant chunk",
            },
            distance=0.4,
        ),
        QueryMatch(
            id="file-b:chunk:0",
            document="budget planning",
            metadata={
                "file_id": "file-b",
                "path": "/tmp/file-b.md",
                "filename": "file-b.md",
                "modality": "text",
                "preview_text": "budget planning",
            },
            distance=0.2,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="semantic retrieval", k=5))

    assert len(response.hits) == 2
    assert response.hits[0].path == "/tmp/file-a.txt"
    assert response.hits[0].preview == "semantic retrieval introduction"


def test_search_service_returns_empty_hits_for_empty_index(tmp_path) -> None:
    settings = make_settings(tmp_path)
    service = SearchService(settings, store=FakeStore([]), embedder=FakeEmbedder())

    response = service.search(SearchRequest(query="anything", k=5))

    assert response.hits == []


def test_search_service_boosts_images_with_query_term_matches(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="pdf-file:chunk:0",
            document="wellness worksheet and planning notes",
            metadata={
                "file_id": "pdf-file",
                "path": "/tmp/wellness.pdf",
                "filename": "wellness.pdf",
                "modality": "pdf",
                "preview_text": "wellness worksheet and planning notes",
            },
            distance=0.15,
        ),
        QueryMatch(
            id="image-file",
            document="A cozy living room with a brightly lit christmas tree and a dark fireplace.",
            metadata={
                "file_id": "image-file",
                "path": "/tmp/christmas_tree.jpeg",
                "filename": "christmas_tree.jpeg",
                "modality": "image",
                "preview_text": "A cozy living room with a brightly lit christmas tree and a dark fireplace.",
                "image_caption": "A cozy living room with a brightly lit christmas tree and a dark fireplace.",
            },
            distance=0.32,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="christmas tree", k=5))

    assert len(response.hits) == 2
    assert response.hits[0].path == "/tmp/christmas_tree.jpeg"
    assert response.hits[0].modality == "image"
    assert response.hits[0].score > response.hits[1].score


def test_search_service_boosts_images_for_related_terms(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="pdf-file:chunk:0",
            document="semester planning and wellness worksheet",
            metadata={
                "file_id": "pdf-file",
                "path": "/tmp/wellness.pdf",
                "filename": "wellness.pdf",
                "modality": "pdf",
                "preview_text": "semester planning and wellness worksheet",
            },
            distance=0.14,
        ),
        QueryMatch(
            id="image-file",
            document="A festive holiday market scene with ornaments, string lights, and a winter square.",
            metadata={
                "file_id": "image-file",
                "path": "/tmp/holiday_market.jpeg",
                "filename": "holiday_market.jpeg",
                "modality": "image",
                "preview_text": "A festive holiday market scene with ornaments, string lights, and a winter square.",
                "image_caption": "A festive holiday market scene with ornaments, string lights, and a winter square.",
            },
            distance=0.30,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="christmas", k=5))

    assert len(response.hits) == 2
    assert response.hits[0].path == "/tmp/holiday_market.jpeg"
    assert response.hits[0].modality == "image"


def test_search_service_diversifies_top_results_with_strong_image_candidate(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="pdf-a:chunk:0",
            document="christmas planning notes",
            metadata={
                "file_id": "pdf-a",
                "path": "/tmp/a.pdf",
                "filename": "a.pdf",
                "modality": "pdf",
                "preview_text": "christmas planning notes",
            },
            distance=0.10,
        ),
        QueryMatch(
            id="pdf-b:chunk:0",
            document="christmas budget notes",
            metadata={
                "file_id": "pdf-b",
                "path": "/tmp/b.pdf",
                "filename": "b.pdf",
                "modality": "pdf",
                "preview_text": "christmas budget notes",
            },
            distance=0.11,
        ),
        QueryMatch(
            id="image-file",
            document="A festive holiday room with ornaments and winter lights.",
            metadata={
                "file_id": "image-file",
                "path": "/tmp/holiday_room.jpeg",
                "filename": "holiday_room.jpeg",
                "modality": "image",
                "preview_text": "A festive holiday room with ornaments and winter lights.",
                "image_caption": "A festive holiday room with ornaments and winter lights.",
            },
            distance=0.18,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="christmas", k=2))

    assert len(response.hits) == 2
    assert any(hit.modality == "image" for hit in response.hits)


def test_search_service_boosts_folder_context_in_second_stage_rerank(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="pdf-file:chunk:0",
            document="generic course notes and study planning",
            metadata={
                "file_id": "pdf-file",
                "path": "/tmp/course-notes.pdf",
                "filename": "course-notes.pdf",
                "modality": "pdf",
                "preview_text": "generic course notes and study planning",
            },
            distance=0.10,
        ),
        QueryMatch(
            id="image-file",
            document="Caption: outdoor market square\nTags: lights, crowd\nConcepts: travel and city scene",
            metadata={
                "file_id": "image-file",
                "path": "/tmp/trips/vienna/market.jpeg",
                "filename": "market.jpeg",
                "modality": "image",
                "preview_text": "outdoor market square",
                "image_caption": "outdoor market square",
                "folder_context": "trips vienna christmas market",
                "folder_path": "trips/vienna/christmas-market",
            },
            distance=0.24,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="vienna christmas market", k=5))

    assert len(response.hits) == 2
    assert response.hits[0].path == "/tmp/trips/vienna/market.jpeg"
    assert response.hits[0].modality == "image"


def test_search_service_rebalances_modalities_after_top_image_hits(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="image-1",
            document="Caption: christmas tree by fireplace\nTags: christmas, tree, fireplace\nConcepts: holiday celebration",
            metadata={
                "file_id": "image-1",
                "path": "/tmp/christmas-tree.jpeg",
                "filename": "christmas-tree.jpeg",
                "modality": "image",
                "preview_text": "christmas tree by fireplace",
                "image_caption": "christmas tree by fireplace",
                "image_tags": "christmas, tree, fireplace",
            },
            distance=0.08,
        ),
        QueryMatch(
            id="image-2",
            document="Caption: holiday lights in living room\nTags: holiday, lights, ornaments\nConcepts: holiday celebration",
            metadata={
                "file_id": "image-2",
                "path": "/tmp/holiday-lights.jpeg",
                "filename": "holiday-lights.jpeg",
                "modality": "image",
                "preview_text": "holiday lights in living room",
                "image_caption": "holiday lights in living room",
                "image_tags": "holiday, lights, ornaments",
            },
            distance=0.10,
        ),
        QueryMatch(
            id="image-3",
            document="Caption: festive market square\nTags: festive, winter, market\nConcepts: travel and city scene",
            metadata={
                "file_id": "image-3",
                "path": "/tmp/festive-market.jpeg",
                "filename": "festive-market.jpeg",
                "modality": "image",
                "preview_text": "festive market square",
                "image_caption": "festive market square",
                "image_tags": "festive, winter, market",
            },
            distance=0.12,
        ),
        QueryMatch(
            id="pdf-1:chunk:0",
            document="christmas planning checklist and holiday notes",
            metadata={
                "file_id": "pdf-1",
                "path": "/tmp/checklist.pdf",
                "filename": "checklist.pdf",
                "modality": "pdf",
                "preview_text": "christmas planning checklist and holiday notes",
            },
            distance=0.14,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="christmas", k=3))

    assert len(response.hits) == 3
    assert response.hits[0].modality == "image"
    assert any(hit.modality == "pdf" for hit in response.hits)


def test_search_service_returns_results_sorted_by_final_score(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="pdf-1:chunk:0",
            document="st mark church summary",
            metadata={
                "file_id": "pdf-1",
                "path": "/tmp/church.pdf",
                "filename": "church.pdf",
                "modality": "pdf",
                "preview_text": "st mark church summary",
            },
            distance=0.30,
        ),
        QueryMatch(
            id="image-1",
            document="Caption: st mark church square\nTags: church, square, zagreb\nConcepts: travel and architecture",
            metadata={
                "file_id": "image-1",
                "path": "/tmp/stmarks.jpeg",
                "filename": "stmarks.jpeg",
                "modality": "image",
                "preview_text": "st mark church square",
                "image_caption": "st mark church square",
                "image_tags": "church, square, zagreb",
            },
            distance=0.34,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="st mark church", k=2))

    assert len(response.hits) == 2
    assert response.hits[0].score >= response.hits[1].score


def test_search_service_penalizes_weak_image_semantic_match_without_field_support(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="image-1",
            document="Caption: golden temple by reflective pond\nTags: golden temple, pavilion, pond, reflection\nConcepts: architecture, nature, travel, culture, spirituality",
            metadata={
                "file_id": "image-1",
                "path": "/tmp/temple.jpeg",
                "filename": "temple.jpeg",
                "modality": "image",
                "preview_text": "golden temple by reflective pond",
                "image_caption": "golden temple by reflective pond",
                "image_tags": "golden temple, pavilion, pond, reflection",
                "image_concepts": "architecture, nature, travel, culture, spirituality",
            },
            distance=0.30,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="st mark church", k=1))

    assert len(response.hits) == 1
    assert 0.60 <= response.hits[0].score <= 0.67


def test_search_service_does_not_treat_mark_as_market_token_match(tmp_path) -> None:
    settings = make_settings(tmp_path)
    matches = [
        QueryMatch(
            id="pdf-1:chunk:0",
            document="The Mark Atlanta move out statement",
            metadata={
                "file_id": "pdf-1",
                "path": "/tmp/the-mark.pdf",
                "filename": "The Mark Atlanta Move-out Statement.pdf",
                "modality": "pdf",
                "preview_text": "The Mark Atlanta move out statement",
            },
            distance=0.18,
        ),
        QueryMatch(
            id="image-1",
            document="Caption: illuminated christmas market in front of city hall\nTags: christmas market, vienna, city hall\nConcepts: christmas, holiday season",
            metadata={
                "file_id": "image-1",
                "path": "/tmp/market.jpeg",
                "filename": "IMG_8451.HEIC",
                "modality": "image",
                "preview_text": "illuminated christmas market in front of city hall",
                "image_caption": "illuminated christmas market in front of city hall",
                "image_tags": "christmas market, vienna, city hall",
                "image_concepts": "christmas, holiday season",
            },
            distance=0.20,
        ),
    ]

    service = SearchService(settings, store=FakeStore(matches), embedder=FakeEmbedder())
    response = service.search(SearchRequest(query="The Mark Atlanta", k=2))

    assert len(response.hits) == 2
    assert response.hits[0].path == "/tmp/the-mark.pdf"
    assert response.hits[0].score > response.hits[1].score
