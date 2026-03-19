from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.models.schemas import HealthResponse, SearchRequest, SearchResponse
from app.services.search import SearchService

router = APIRouter()


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(get_settings())


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        collection=settings.chroma_collection,
        input_dir=str(settings.input_dir),
    )


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    try:
        return service.search(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
