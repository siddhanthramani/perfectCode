"""
Abstract base for indexer adapters.

An indexer stores QualityEvents in a searchable form — full-text, vector
embeddings, graph structures, or agentic pipelines. All implementations
expose the same three operations: index, search, similar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from perfectcode.core.models import (
    IndexEntry,
    QualityEvent,
    SearchQuery,
    SearchResult,
)


class IndexerAdapter(ABC):
    """
    Common interface over any search / retrieval backend.

    Implementations may wrap SQLite FTS, a vector DB, a graph database,
    or an agentic orchestration that calls other tools.
    """

    @property
    @abstractmethod
    def indexer_name(self) -> str:
        """Unique name for this indexer instance, e.g. ``'sqlite_fts'``."""
        ...

    @abstractmethod
    async def index(self, events: list[QualityEvent]) -> list[IndexEntry]:
        """
        Insert or update events in the index.

        Returns an IndexEntry per event describing what was stored.
        """
        ...

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Search the index and return ranked results.

        The query may carry free text, a code snippet, tag filters, and
        metadata constraints.
        """
        ...

    @abstractmethod
    async def similar(self, reference: QualityEvent | str) -> list[SearchResult]:
        """
        Find events similar to *reference*.

        *reference* can be a full QualityEvent (uses its content) or a raw
        string (code snippet, error message, etc.).
        """
        ...

    @abstractmethod
    async def delete(self, event_ids: list[str]) -> int:
        """Remove events from the index. Returns number of entries deleted."""
        ...
