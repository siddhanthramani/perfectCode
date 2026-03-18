"""
Abstract base for reranker adapters.

Rerankers refine and reorder results produced by IndexerAdapters. They are
a separate pipeline stage: indexers retrieve, rerankers refine. This follows
the pattern established by Haystack (Retriever vs Ranker), LlamaIndex
(Retriever vs NodePostprocessor), and Elasticsearch (query vs rescore).

A class may implement both IndexerAdapter and RerankerAdapter when it can
do both (e.g., the taxonomy navigates the tag tree for retrieval *and*
boosts tag-relevant results from other indexers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from perfectcode.core.models import SearchQuery, SearchResult


class RerankerAdapter(ABC):
    """
    Refine and reorder search results produced by one or more indexers.

    Implementations may use tag overlap, embedding similarity, cross-encoder
    models, rule engines, or any other signal to rescore results.
    """

    @property
    @abstractmethod
    def reranker_name(self) -> str:
        """Identifier for this reranker, e.g. ``'taxonomy'``, ``'cross_encoder'``."""
        ...

    @abstractmethod
    async def rerank(
        self,
        results: list[SearchResult],
        query: SearchQuery,
    ) -> list[SearchResult]:
        """
        Reorder and optionally filter *results* for *query*.

        Implementations should return a new list sorted by their refined
        scores. They may adjust ``SearchResult.score``, drop low-confidence
        results, or inject additional metadata (e.g., ``matched_tags``).
        """
        ...
