from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_embedding_model: str | None = Field(
        default=None,
        alias="GEMINI_EMBEDDING_MODEL",
    )
    gemini_vision_model: str = Field(
        default="gemini-2.5-flash",
        alias="GEMINI_VISION_MODEL",
    )
    input_dir: Path = Field(default=Path("data/input"), alias="INPUT_DIR")
    chroma_dir: Path = Field(default=Path("data/chroma"), alias="CHROMA_DIR")
    thumbs_dir: Path = Field(default=Path("data/thumbs"), alias="THUMBS_DIR")
    chroma_collection: str = Field(default="macmemory", alias="CHROMA_COLLECTION")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
