"""
Vector-based indexer and reranker using embedding similarity.

Default embedding is a dependency-free bag-of-words approach (term frequency
vectors with cosine similarity). Users can swap in real models (sentence-
transformers, OpenAI embeddings, etc.) by providing a custom ``embed_fn``
to the constructor.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

from perfectcode.adapters.indexers.base import IndexerAdapter
from perfectcode.adapters.indexers.reranker import RerankerAdapter
from perfectcode.core.models import (
    IndexEntry,
    QualityEvent,
    SearchQuery,
    SearchResult,
)

EmbedFn = Callable[[str], list[float]]


# ---------------------------------------------------------------------------
# Default bag-of-words embedder (zero external dependencies)
# ---------------------------------------------------------------------------

_VOCAB: dict[str, int] = {}


def _bow_embed(text: str) -> list[float]:
    """
    Bag-of-words embedding: builds a shared vocabulary on the fly and returns
    a term-frequency vector. Suitable for dev/test; swap with a real model
    for production.
    """
    tokens = text.lower().split()
    for token in tokens:
        if token not in _VOCAB:
            _VOCAB[token] = len(_VOCAB)

    vec = [0.0] * len(_VOCAB)
    counts = Counter(tokens)
    for token, count in counts.items():
        vec[_VOCAB[token]] = float(count)
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, handling length mismatches."""
    length = max(len(a), len(b))
    if length == 0:
        return 0.0

    a_ext = a + [0.0] * (length - len(a))
    b_ext = b + [0.0] * (length - len(b))

    dot = sum(x * y for x, y in zip(a_ext, b_ext))
    mag_a = math.sqrt(sum(x * x for x in a_ext))
    mag_b = math.sqrt(sum(x * x for x in b_ext))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _event_to_text(event: QualityEvent) -> str:
    parts = [event.title, event.description]
    parts.extend(event.stack_traces)
    for loc in event.code_locations:
        parts.append(loc.file_path)
        if loc.function_name:
            parts.append(loc.function_name)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class VectorIndexerAdapter(IndexerAdapter, RerankerAdapter):
    """
    Embedding-based search over QualityEvents.

    As an **IndexerAdapter**: embeds events on index, embeds queries on search,
    and returns results ranked by cosine similarity.

    As a **RerankerAdapter**: re-scores results from other indexers by computing
    cosine similarity between the query embedding and each result's event
    embedding, blending with the original score.

    The embedding function is pluggable. Default is a bag-of-words approach
    that requires no external dependencies. To use a real model::

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        adapter = VectorIndexerAdapter(embed_fn=lambda text: model.encode(text).tolist())
    """

    def __init__(self, embed_fn: EmbedFn | None = None) -> None:
        self._embed = embed_fn or _bow_embed
        self._store: dict[str, tuple[QualityEvent, list[float]]] = {}

    @property
    def indexer_name(self) -> str:
        return "vector"

    @property
    def reranker_name(self) -> str:
        return "vector"

    # ------------------------------------------------------------------
    # IndexerAdapter — retrieval
    # ------------------------------------------------------------------

    async def index(self, events: list[QualityEvent]) -> list[IndexEntry]:
        entries: list[IndexEntry] = []
        for event in events:
            text = _event_to_text(event)
            embedding = self._embed(text)
            self._store[event.id] = (event, embedding)
            entries.append(
                IndexEntry(
                    event_id=event.id,
                    indexer_name=self.indexer_name,
                    index_data={"dimensions": len(embedding)},
                )
            )
        return entries

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        query_text = f"{query.text} {query.code_snippet}".strip()
        if not query_text:
            return []

        query_vec = self._embed(query_text)
        return self._rank_against(query_vec, query.max_results, query.min_score)

    async def similar(self, reference: QualityEvent | str) -> list[SearchResult]:
        if isinstance(reference, QualityEvent):
            if reference.id in self._store:
                ref_vec = self._store[reference.id][1]
            else:
                ref_vec = self._embed(_event_to_text(reference))
            exclude_id = reference.id
        else:
            ref_vec = self._embed(reference)
            exclude_id = None

        results = self._rank_against(ref_vec, max_results=5, min_score=0.1)
        if exclude_id:
            results = [r for r in results if r.event.id != exclude_id]
        return results

    async def delete(self, event_ids: list[str]) -> int:
        deleted = 0
        for eid in event_ids:
            if eid in self._store:
                del self._store[eid]
                deleted += 1
        return deleted

    # ------------------------------------------------------------------
    # RerankerAdapter — embedding-based reranking
    # ------------------------------------------------------------------

    async def rerank(
        self,
        results: list[SearchResult],
        query: SearchQuery,
    ) -> list[SearchResult]:
        if not results:
            return results

        query_text = f"{query.text} {query.code_snippet}".strip()
        if not query_text:
            return results

        query_vec = self._embed(query_text)

        reranked: list[SearchResult] = []
        for result in results:
            stored = self._store.get(result.event.id)
            if stored:
                event_vec = stored[1]
            else:
                event_vec = self._embed(_event_to_text(result.event))

            vec_score = _cosine_similarity(query_vec, event_vec)
            blended = 0.5 * result.score + 0.5 * vec_score

            reranked.append(
                SearchResult(
                    event=result.event,
                    score=blended,
                    indexer_name=result.indexer_name,
                    matched_tags=result.matched_tags,
                    explanation=result.explanation,
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rank_against(
        self,
        query_vec: list[float],
        max_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        scored: list[SearchResult] = []
        for event, event_vec in self._store.values():
            score = _cosine_similarity(query_vec, event_vec)
            if score < min_score:
                continue
            scored.append(
                SearchResult(event=event, score=score, indexer_name=self.indexer_name)
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:max_results]
