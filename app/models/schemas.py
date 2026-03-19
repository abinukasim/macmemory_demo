from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MetadataValue = str | int | float | bool
Modality = Literal["text", "image", "pdf"]


class IndexRecord(BaseModel):
    id: str
    file_id: str
    path: str
    modality: Modality
    document: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class IndexedAsset(IndexRecord):
    embedding: list[float]


class QueryMatch(BaseModel):
    id: str
    document: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    distance: float


class IndexingSummary(BaseModel):
    input_dir: str
    scanned: int
    supported: int
    skipped: int
    indexed: int = 0
    text_chunks: int = 0
    pdf_chunks: int = 0
    image_files: int = 0
    failed: int = 0
    duration_seconds: float = 0.0


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=25)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class SearchHit(BaseModel):
    path: str
    modality: Modality
    score: float
    preview: str | None = None
    thumbnail_path: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    collection: str
    input_dir: str
