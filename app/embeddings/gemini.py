import logging
import re
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from app.core.constants import EMBEDDING_BATCH_SIZE
from app.core.config import Settings
from app.utils.images import normalize_image_for_embedding

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageDescription:
    caption: str
    tags: list[str]
    concepts: list[str]

    def to_index_text(self) -> str:
        lines = [f"Caption: {self.caption}"]
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        if self.concepts:
            lines.append(f"Concepts: {', '.join(self.concepts)}")
        return "\n".join(lines)


class GeminiEmbedder:
    """Gemini embedding adapter for text queries, documents, and images."""

    def __init__(self, settings: Settings, client: genai.Client | None = None) -> None:
        self.settings = settings
        self._client = client
        self._captioning_disabled_reason: str | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._require_configuration()
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def embed_query(self, query: str) -> list[float]:
        self._require_configuration()
        response = self.client.models.embed_content(
            model=self.settings.gemini_embedding_model,
            contents=query,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return self._extract_single_embedding(response)

    def embed_documents(self, contents: list[str], title: str | None = None) -> list[list[float]]:
        self._require_configuration()
        if not contents:
            return []

        config: dict[str, str] = {"task_type": "RETRIEVAL_DOCUMENT"}
        if title:
            config["title"] = title

        embeddings: list[list[float]] = []
        for batch in _batched(contents, EMBEDDING_BATCH_SIZE):
            response = self.client.models.embed_content(
                model=self.settings.gemini_embedding_model,
                contents=batch,
                config=config,
            )
            embeddings.extend([embedding.values or [] for embedding in response.embeddings or []])

        if len(embeddings) != len(contents):
            raise RuntimeError("Gemini returned an unexpected number of document embeddings.")
        return embeddings

    def embed_image(self, path: Path) -> list[float]:
        description = self.describe_image(path)
        embeddings = self.embed_documents([description.to_index_text()], title=path.name)
        if not embeddings:
            raise RuntimeError(f"Image embedding failed for {path.name}.")
        return embeddings[0]

    def describe_image(self, path: Path) -> ImageDescription:
        self._require_configuration()
        if self._captioning_disabled_reason:
            return self._fallback_image_description(path)

        image_bytes, mime_type = normalize_image_for_embedding(path)

        prompt = (
            "You are generating retrieval metadata for local image search.\n"
            "Return exactly three lines in this format:\n"
            "caption: <one short sentence>\n"
            "tags: <8-15 concise comma-separated tags>\n"
            "concepts: <3-6 broader comma-separated concepts>\n"
            "Rules:\n"
            "- Prefer concrete nouns and scene descriptors.\n"
            "- Include likely event/theme words when supported by the image.\n"
            "- Avoid uncertain claims and avoid mentioning that this is an image.\n"
            "- Keep the caption under 24 words."
        )

        try:
            response = self.client.models.generate_content(
                model=self.settings.gemini_vision_model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        ]
                    )
                ],
            )
        except Exception as exc:  # pragma: no cover - depends on remote provider support
            message = str(exc)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                self._captioning_disabled_reason = message
                logger.warning(
                    "Image captioning quota exhausted; disabling vision captions for the rest of this indexing run."
                )
            logger.warning("Image captioning failed for %s; falling back to filename-based caption: %s", path.name, exc)
            return self._fallback_image_description(path)

        description = _parse_image_description((response.text or "").strip())
        if description is not None:
            return description
        return self._fallback_image_description(path, raw_text=(response.text or "").strip())

    def caption_image(self, path: Path) -> str:
        return self.describe_image(path).caption

    def embed_pdf(self, path: Path) -> list[float]:
        self._require_configuration()

        try:
            response = self.client.models.embed_content(
                model=self.settings.gemini_embedding_model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=path.read_bytes(), mime_type="application/pdf"),
                        ]
                    )
                ],
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
        except Exception as exc:  # pragma: no cover - depends on remote provider support
            raise RuntimeError(
                f"Native PDF embedding failed for {path.name}. Confirm the configured Gemini model/provider supports direct PDF embeddings."
            ) from exc

        return self._extract_single_embedding(response)

    def _extract_single_embedding(self, response: types.EmbedContentResponse) -> list[float]:
        embeddings = response.embeddings or []
        if not embeddings or not embeddings[0].values:
            raise RuntimeError("Gemini returned an empty embedding response.")
        return embeddings[0].values

    def _require_configuration(self) -> None:
        if not self.settings.gemini_api_key:
            raise RuntimeError("Set GEMINI_API_KEY before generating embeddings.")
        if not self.settings.gemini_embedding_model:
            raise RuntimeError("Set GEMINI_EMBEDDING_MODEL before generating embeddings.")

    def _fallback_image_caption(self, path: Path) -> str:
        stem = path.stem.replace("-", " ").replace("_", " ").strip()
        if not stem:
            return f"photo {path.suffix.lower().lstrip('.')}"
        if re.fullmatch(r"IMG\s*\d+", stem, re.IGNORECASE):
            return "personal photo from camera roll"
        if re.fullmatch(r"[A-F0-9-]{12,}", stem, re.IGNORECASE):
            return "photo with autogenerated filename"
        return stem

    def _fallback_image_description(self, path: Path, raw_text: str = "") -> ImageDescription:
        caption = raw_text.strip() or self._fallback_image_caption(path)
        tags = _extract_terms(caption, limit=12)
        concepts = _derive_concepts(tags)
        return ImageDescription(caption=caption, tags=tags, concepts=concepts)


def _batched(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _parse_image_description(raw_text: str) -> ImageDescription | None:
    if not raw_text:
        return None

    cleaned = raw_text.replace("```", "").strip()
    fields: dict[str, str] = {}
    for line in cleaned.splitlines():
        match = re.match(r"^\s*(caption|tags|concepts)\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
        if match:
            fields[match.group(1).lower()] = match.group(2).strip()

    caption = fields.get("caption", "").strip()
    tags = _split_terms(fields.get("tags", ""))
    concepts = _split_terms(fields.get("concepts", ""))
    if not caption:
        return None

    if not tags:
        tags = _extract_terms(caption, limit=12)
    if not concepts:
        concepts = _derive_concepts(tags)

    return ImageDescription(caption=caption, tags=tags, concepts=concepts)


def _split_terms(value: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;/]", value):
        normalized = " ".join(part.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _extract_terms(text: str, limit: int) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "the",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "while",
        "near",
        "beside",
        "next",
        "through",
        "shows",
        "showing",
        "photo",
        "image",
    }
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) < 3 or token in stopwords or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _derive_concepts(tags: list[str]) -> list[str]:
    concept_map = {
        "christmas": "holiday celebration",
        "holiday": "holiday celebration",
        "festive": "holiday celebration",
        "market": "travel and city scene",
        "church": "travel and architecture",
        "temple": "travel and architecture",
        "lake": "nature and water",
        "mountain": "nature and landscape",
        "garden": "nature and plants",
        "flowers": "nature and plants",
        "food": "dining and meals",
        "burger": "dining and meals",
        "coffee": "cafe and drinks",
        "menu": "screen or signage",
        "application": "screen or software",
        "screenshot": "screen or software",
        "soccer": "sports and activity",
        "football": "sports and activity",
        "students": "campus and daily life",
        "campus": "campus and daily life",
        "fireplace": "indoor decor",
        "living": "indoor decor",
    }
    concepts: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        concept = concept_map.get(tag)
        if concept and concept not in seen:
            seen.add(concept)
            concepts.append(concept)
        if len(concepts) >= 6:
            break
    return concepts
