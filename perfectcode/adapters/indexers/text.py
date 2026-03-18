"""
Text-based indexer and reranker using in-memory substring matching.

This is the v0 reference implementation. A production version would use
SQLite FTS5, Elasticsearch, Meilisearch, or similar.
"""

from __future__ import annotations

from perfectcode.adapters.indexers.base import IndexerAdapter
from perfectcode.adapters.indexers.reranker import RerankerAdapter
from perfectcode.core.models import (
    IndexEntry,
    QualityEvent,
    SearchQuery,
    SearchResult,
)


class TextIndexerAdapter(IndexerAdapter, RerankerAdapter):
    """
    In-memory text search over QualityEvents.

    As an **IndexerAdapter**: stores events and searches by checking whether
    query terms appear in the title, description, or stack traces.

    As a **RerankerAdapter**: re-scores results from other indexers (vector,
    taxonomy, etc.) using term-match relevance, so that keyword-relevant
    results float to the top regardless of which indexer found them.
    """

    def __init__(self) -> None:
        self._store: dict[str, QualityEvent] = {}

    @property
    def indexer_name(self) -> str:
        return "text_inmemory"

    @property
    def reranker_name(self) -> str:
        return "text_inmemory"

    # ------------------------------------------------------------------
    # IndexerAdapter — retrieval
    # ------------------------------------------------------------------

    async def index(self, events: list[QualityEvent]) -> list[IndexEntry]:
        entries: list[IndexEntry] = []
        for event in events:
            self._store[event.id] = event
            entries.append(
                IndexEntry(
                    event_id=event.id,
                    indexer_name=self.indexer_name,
                    index_data={"stored": True},
                )
            )
        return entries

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        terms = self._extract_terms(query)
        if not terms:
            return []

        results: list[SearchResult] = []
        for event in self._store.values():
            score = self._term_score(event, terms)
            if score < query.min_score:
                continue
            results.append(
                SearchResult(event=event, score=score, indexer_name=self.indexer_name)
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.max_results]

    async def similar(self, reference: QualityEvent | str) -> list[SearchResult]:
        if isinstance(reference, QualityEvent):
            text = f"{reference.title} {reference.description}"
        else:
            text = reference
        return await self.search(SearchQuery(text=text, max_results=5, min_score=0.1))

    async def delete(self, event_ids: list[str]) -> int:
        deleted = 0
        for eid in event_ids:
            if eid in self._store:
                del self._store[eid]
                deleted += 1
        return deleted

    # ------------------------------------------------------------------
    # RerankerAdapter — term-match reranking
    # ------------------------------------------------------------------

    async def rerank(
        self,
        results: list[SearchResult],
        query: SearchQuery,
    ) -> list[SearchResult]:
        terms = self._extract_terms(query)
        if not terms or not results:
            return results

        reranked: list[SearchResult] = []
        for result in results:
            text_score = self._term_score(result.event, terms)
            blended = 0.6 * result.score + 0.4 * text_score
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

    @staticmethod
    def _extract_terms(query: SearchQuery) -> list[str]:
        raw = f"{query.text} {query.code_snippet}".lower().split()
        return [t for t in raw if len(t) > 1]

    @staticmethod
    def _term_score(event: QualityEvent, terms: list[str]) -> float:
        corpus = f"{event.title} {event.description} {' '.join(event.stack_traces)}".lower()
        hits = sum(1 for t in terms if t in corpus)
        return hits / len(terms) if terms else 0.0
