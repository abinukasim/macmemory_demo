import logging
import re
import time

from app.core.config import Settings
from app.embeddings.gemini import GeminiEmbedder
from app.models.schemas import QueryMatch, SearchHit, SearchRequest, SearchResponse
from app.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

TERM_EXPANSIONS = {
    "christmas": {
        "holiday",
        "holidays",
        "festive",
        "festival",
        "winter",
        "ornament",
        "ornaments",
        "tree",
        "lights",
        "gift",
        "gifts",
        "present",
        "presents",
        "fireplace",
        "garland",
        "wreath",
        "market",
    },
    "holiday": {
        "christmas",
        "festive",
        "festival",
        "winter",
        "ornament",
        "lights",
        "gift",
        "gifts",
        "present",
        "presents",
        "market",
    },
    "holidays": {
        "christmas",
        "holiday",
        "festive",
        "festival",
        "winter",
        "lights",
        "gifts",
        "presents",
    },
    "festive": {
        "christmas",
        "holiday",
        "festival",
        "ornament",
        "lights",
        "winter",
    },
    "winter": {
        "christmas",
        "holiday",
        "festive",
        "snow",
        "lights",
        "market",
    },
    "mountain": {"lake", "hike", "trail", "summit", "overlook", "forest"},
    "lake": {"mountain", "water", "shore", "reflection", "boat", "park"},
    "resume": {"cv", "career", "job", "work", "experience", "education"},
    "wellness": {"health", "stress", "mindfulness", "exercise", "sleep"},
}


class SearchService:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore | None = None,
        embedder: GeminiEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or ChromaStore(settings)
        self.embedder = embedder or GeminiEmbedder(settings)

    def search(self, request: SearchRequest) -> SearchResponse:
        started_at = time.perf_counter()
        total_records = self.store.count()
        if total_records == 0:
            logger.info("Search completed: query=%r hits=0 duration=%.3fs", request.query, time.perf_counter() - started_at)
            return SearchResponse(query=request.query, hits=[])

        query_embedding = self.embedder.embed_query(request.query)
        raw_limit = min(total_records, max(request.k * 25, 200))
        raw_matches = self.store.query(query_embedding, limit=raw_limit)
        response = SearchResponse(
            query=request.query,
            hits=self._aggregate_hits(raw_matches, request.query, request.k),
        )
        logger.info(
            "Search completed: query=%r hits=%s duration=%.3fs",
            request.query,
            len(response.hits),
            time.perf_counter() - started_at,
        )
        return response

    def _aggregate_hits(self, matches: list[QueryMatch], query: str, limit: int) -> list[SearchHit]:
        best_by_file: dict[str, SearchHit] = {}
        query_terms = _tokenize_query(query)
        query_phrase = query.strip().lower()

        for match in matches:
            metadata = dict(match.metadata)
            file_id = str(metadata.get("file_id", match.id))
            base_score = max(0.0, 1.0 - max(match.distance, 0.0))
            preview = str(metadata.get("preview_text", "")).strip() or str(metadata.get("filename", ""))
            thumbnail_path = str(metadata.get("thumbnail_path", "")).strip() or None
            modality = str(metadata.get("modality", "text"))
            search_text = _build_search_text(match, metadata)
            score = _rerank_score(
                base_score=base_score,
                modality=modality,
                search_text=search_text,
                query_terms=query_terms,
                query_phrase=query_phrase,
            )

            candidate = SearchHit(
                path=str(metadata.get("path", "")),
                modality=modality,
                score=score,
                preview=preview or None,
                thumbnail_path=thumbnail_path,
                metadata=metadata,
            )

            existing = best_by_file.get(file_id)
            if existing is None or candidate.score > existing.score:
                best_by_file[file_id] = candidate

        ranked_hits = sorted(best_by_file.values(), key=lambda hit: hit.score, reverse=True)
        return _diversify_hits(ranked_hits, query, limit)


def _build_search_text(match: QueryMatch, metadata: dict[str, object]) -> str:
    fields = [
        _normalized_field_value("filename", metadata),
        _normalized_field_value("folder_context", metadata),
        _normalized_field_value("folder_path", metadata),
        _normalized_field_value("preview_text", metadata),
        _normalized_field_value("image_caption", metadata),
        _normalized_field_value("image_tags", metadata),
        _normalized_field_value("image_concepts", metadata),
        _normalized_field_value("ocr_text", metadata),
        match.document or "",
    ]
    return " ".join(part for part in fields if part).lower()


def _rerank_score(
    *,
    base_score: float,
    modality: str,
    search_text: str,
    query_terms: list[str],
    query_phrase: str,
) -> float:
    if not search_text or not query_terms:
        return base_score

    matched_terms = {term for term in query_terms if term in search_text}
    expanded_terms = _expand_query_terms(query_terms) - set(query_terms)
    matched_expanded_terms = {term for term in expanded_terms if term in search_text}
    if not matched_terms and not matched_expanded_terms:
        return base_score

    overlap_ratio = len(matched_terms) / len(query_terms)
    expanded_ratio = len(matched_expanded_terms) / max(len(expanded_terms), 1)
    phrase_match = bool(query_phrase and len(query_phrase) >= 3 and query_phrase in search_text)

    boost = 0.0
    boost += 0.10 * overlap_ratio
    boost += 0.05 * expanded_ratio
    if phrase_match:
        boost += 0.08

    if modality == "image":
        boost += 0.05
        boost += 0.09 * overlap_ratio
        boost += 0.07 * expanded_ratio
        if phrase_match:
            boost += 0.04

    return min(0.999, base_score + boost)


def _tokenize_query(query: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 3]


def _expand_query_terms(query_terms: list[str]) -> set[str]:
    expanded: set[str] = set()
    for term in query_terms:
        expanded.update(TERM_EXPANSIONS.get(term, set()))
    return expanded


def _diversify_hits(hits: list[SearchHit], query: str, limit: int) -> list[SearchHit]:
    top_hits = hits[:limit]
    if len(top_hits) < limit or any(hit.modality == "image" for hit in top_hits):
        return _rebalance_modalities(_second_stage_rerank(top_hits, query, limit), limit)

    image_candidates = [hit for hit in hits if hit.modality == "image"]
    if not image_candidates:
        return _rebalance_modalities(_second_stage_rerank(top_hits, query, limit), limit)

    best_image = image_candidates[0]
    last_score = top_hits[-1].score
    if best_image.score + 0.12 < last_score:
        return _rebalance_modalities(_second_stage_rerank(top_hits, query, limit), limit)

    diversified = top_hits[:-1] + [best_image]
    diversified = sorted(diversified, key=lambda hit: hit.score, reverse=True)
    return _rebalance_modalities(_second_stage_rerank(diversified, query, limit), limit)


def _second_stage_rerank(hits: list[SearchHit], query: str, limit: int) -> list[SearchHit]:
    if not hits:
        return []

    query_terms = _tokenize_query(query)
    query_phrase = query.strip().lower()
    reranked_hits: list[SearchHit] = []
    for hit in hits[: max(limit * 4, limit)]:
        second_stage_score = min(
            0.999,
            hit.score + _field_match_bonus(hit, query_terms, query_phrase),
        )
        reranked_hits.append(hit.model_copy(update={"score": second_stage_score}))
    return sorted(reranked_hits, key=lambda hit: hit.score, reverse=True)[:limit]


def _field_match_bonus(hit: SearchHit, query_terms: list[str], query_phrase: str) -> float:
    if not query_terms:
        return 0.0

    metadata = hit.metadata
    field_weights = {
        "filename": 0.10,
        "folder_context": 0.14,
        "preview_text": 0.14,
        "image_caption": 0.14,
        "image_tags": 0.16,
        "image_concepts": 0.12,
        "ocr_text": 0.18,
    }

    bonus = 0.0
    expanded_terms = _expand_query_terms(query_terms) - set(query_terms)
    specific_match_found = False
    for field, weight in field_weights.items():
        value = _normalized_field_value(field, metadata).lower()
        if not value:
            continue
        value_tokens = set(_tokenize_query(value))
        matched_terms = set(query_terms) & value_tokens
        matched_expanded_terms = expanded_terms & value_tokens
        if matched_terms:
            bonus += weight * (len(matched_terms) / len(query_terms))
            if field != "image_concepts":
                specific_match_found = True
        if matched_expanded_terms:
            expansion_weight = 0.7 if hit.modality == "image" else 0.5
            bonus += (weight * expansion_weight) * (len(matched_expanded_terms) / max(len(expanded_terms), 1))
            if field != "image_concepts":
                specific_match_found = True
        if query_phrase and len(query_phrase) >= 3 and query_phrase in value:
            bonus += weight * 0.4
            if field != "image_concepts":
                specific_match_found = True

    if hit.modality == "image":
        folder_context = _normalized_field_value("folder_context", metadata)
        ocr_text = _normalized_field_value("ocr_text", metadata)
        folder_tokens = set(_tokenize_query(folder_context))
        ocr_tokens = set(_tokenize_query(ocr_text))
        if folder_context and (set(query_terms) & folder_tokens):
            bonus += 0.005
        if ocr_text and (set(query_terms) & ocr_tokens):
            bonus += 0.01
        if len(query_terms) >= 2 and not specific_match_found:
            bonus -= 0.06
        if str(metadata.get("embedding_kind", "")) == "ocr" and not ocr_text:
            bonus -= 0.04

    return bonus


def _rebalance_modalities(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    if len(hits) <= 2:
        return hits[:limit]

    locked = hits[:1]
    remaining = hits[len(locked) :]
    selected = list(locked)
    counts: dict[str, int] = {}
    for hit in selected:
        counts[hit.modality] = counts.get(hit.modality, 0) + 1

    while remaining and len(selected) < limit:
        best_index = 0
        best_score = float("-inf")
        for index, hit in enumerate(remaining):
            penalty = 0.06 * counts.get(hit.modality, 0)
            adjusted = hit.score - penalty
            if adjusted > best_score:
                best_score = adjusted
                best_index = index
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        counts[chosen.modality] = counts.get(chosen.modality, 0) + 1

    if not any(hit.modality == "image" for hit in selected):
        image_candidates = [hit for hit in hits if hit.modality == "image"]
        if image_candidates and selected:
            best_image = image_candidates[0]
            if best_image.score + 0.06 >= selected[-1].score:
                selected[-1] = best_image

    return sorted(selected[:limit], key=lambda hit: hit.score, reverse=True)


def _normalized_field_value(field: str, metadata: dict[str, object]) -> str:
    value = str(metadata.get(field, "")).strip()
    if field == "ocr_text" and value and _is_low_quality_ocr(value):
        return ""
    return value


def _is_low_quality_ocr(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text)
    if len(tokens) < 6:
        return True
    long_tokens = [token for token in tokens if len(token) >= 4]
    vowel_tokens = [token for token in tokens if re.search(r"[aeiouAEIOU]", token)]
    return (len(long_tokens) / len(tokens) < 0.35) or (len(vowel_tokens) / len(tokens) < 0.55)
