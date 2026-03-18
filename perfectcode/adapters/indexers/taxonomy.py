"""
Taxonomy-based indexer and reranker.

Wraps ``QualityTaxonomyEngine`` behind both ``IndexerAdapter`` (for
tree-navigated retrieval) and ``RerankerAdapter`` (for tag-based boosting
of results produced by other indexers).

Retrieval: keyword-matches query text against tag names and descriptions,
collects events under matching branches, and returns them as SearchResults.

Reranking: given merged results from all indexers, boosts events whose tags
overlap with the query's tag filter or with automatically inferred relevant
branches, and demotes untagged events.
"""

from __future__ import annotations

import logging

from perfectcode.adapters.indexers.base import IndexerAdapter
from perfectcode.adapters.indexers.reranker import RerankerAdapter
from perfectcode.core.models import (
    IndexEntry,
    QualityEvent,
    SearchQuery,
    SearchResult,
)
from perfectcode.core.taxonomy import QualityTaxonomyEngine

logger = logging.getLogger(__name__)

_TAG_BOOST = 0.3
_UNTAGGED_PENALTY = 0.5


class TaxonomyIndexerAdapter(IndexerAdapter, RerankerAdapter):
    """
    Uses the hierarchical tag tree for both retrieval and reranking.

    As an **IndexerAdapter**: navigates the tag tree to find events whose tags
    match query terms, returning them ranked by depth-of-match.

    As a **RerankerAdapter**: boosts results from other indexers when their
    events carry tags relevant to the query, and penalises untagged events.
    """

    def __init__(self, taxonomy: QualityTaxonomyEngine) -> None:
        self._taxonomy = taxonomy
        self._events: dict[str, QualityEvent] = {}

    @property
    def indexer_name(self) -> str:
        return "taxonomy"

    @property
    def reranker_name(self) -> str:
        return "taxonomy"

    # ------------------------------------------------------------------
    # IndexerAdapter — retrieval
    # ------------------------------------------------------------------

    async def index(self, events: list[QualityEvent]) -> list[IndexEntry]:
        entries: list[IndexEntry] = []
        for event in events:
            self._events[event.id] = event
            entries.append(
                IndexEntry(
                    event_id=event.id,
                    indexer_name=self.indexer_name,
                    index_data={"tagged": True},
                )
            )
        return entries

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        matching_tag_ids = self._match_tags(query)
        if not matching_tag_ids:
            return []

        event_ids: set[str] = set()
        for tag_id in matching_tag_ids:
            event_ids.update(
                self._taxonomy.get_events_for_tag(tag_id, include_subtree=True)
            )

        results: list[SearchResult] = []
        for eid in event_ids:
            event = self._events.get(eid)
            if not event:
                continue
            tags = self._taxonomy.get_event_tags(eid)
            tag_paths = [t.path for t in tags]
            depth_score = max(
                (t.path.count("/") + 1) / 5.0 for t in tags
            ) if tags else 0.1
            score = min(depth_score, 1.0)
            if score < query.min_score:
                continue
            results.append(
                SearchResult(
                    event=event,
                    score=score,
                    indexer_name=self.indexer_name,
                    matched_tags=tag_paths,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.max_results]

    async def similar(self, reference: QualityEvent | str) -> list[SearchResult]:
        if isinstance(reference, QualityEvent):
            tags = self._taxonomy.get_event_tags(reference.id)
            if not tags:
                return []
            sibling_event_ids: set[str] = set()
            for tag in tags:
                if tag.parent_id:
                    for sibling_tag in self._taxonomy.get_children(tag.parent_id):
                        sibling_event_ids.update(
                            self._taxonomy.get_events_for_tag(sibling_tag.id)
                        )
                sibling_event_ids.update(
                    self._taxonomy.get_events_for_tag(tag.id, include_subtree=True)
                )
            sibling_event_ids.discard(reference.id)

            results: list[SearchResult] = []
            ref_tag_ids = {t.id for t in tags}
            for eid in sibling_event_ids:
                event = self._events.get(eid)
                if not event:
                    continue
                other_tags = self._taxonomy.get_event_tags(eid)
                other_tag_ids = {t.id for t in other_tags}
                overlap = len(ref_tag_ids & other_tag_ids)
                score = overlap / max(len(ref_tag_ids | other_tag_ids), 1)
                results.append(
                    SearchResult(
                        event=event,
                        score=score,
                        indexer_name=self.indexer_name,
                        matched_tags=[t.path for t in other_tags],
                    )
                )
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:5]
        else:
            query = SearchQuery(text=reference, max_results=5, min_score=0.1)
            return await self.search(query)

    async def delete(self, event_ids: list[str]) -> int:
        deleted = 0
        for eid in event_ids:
            if eid in self._events:
                del self._events[eid]
                deleted += 1
        return deleted

    # ------------------------------------------------------------------
    # RerankerAdapter — boost/demote based on tags
    # ------------------------------------------------------------------

    async def rerank(
        self,
        results: list[SearchResult],
        query: SearchQuery,
    ) -> list[SearchResult]:
        if not results:
            return results

        relevant_tag_ids = set[str]()

        if query.tag_filter:
            relevant_tag_ids.update(query.tag_filter.tag_ids)
            for path in query.tag_filter.tag_paths:
                node = self._taxonomy.find_by_path(path)
                if node:
                    relevant_tag_ids.add(node.id)

        if not relevant_tag_ids:
            relevant_tag_ids = {
                tid for tid in self._match_tags(query)
            }

        reranked: list[SearchResult] = []
        for result in results:
            event_tags = self._taxonomy.get_event_tags(result.event.id)
            event_tag_ids = {t.id for t in event_tags}
            tag_paths = [t.path for t in event_tags]

            adjusted_score = result.score
            if relevant_tag_ids and event_tag_ids:
                overlap = len(relevant_tag_ids & event_tag_ids)
                if overlap > 0:
                    adjusted_score += _TAG_BOOST * overlap
            elif not event_tag_ids:
                adjusted_score *= _UNTAGGED_PENALTY

            reranked.append(
                SearchResult(
                    event=result.event,
                    score=adjusted_score,
                    indexer_name=result.indexer_name,
                    matched_tags=tag_paths or result.matched_tags,
                    explanation=result.explanation,
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_tags(self, query: SearchQuery) -> set[str]:
        """Find tag IDs whose name or description matches query terms."""
        text = f"{query.text} {query.code_snippet}".lower().split()
        if not text:
            return set()

        matched: set[str] = set()
        for node in self._taxonomy._nodes.values():
            corpus = f"{node.name} {node.description}".lower()
            if any(term in corpus for term in text):
                matched.add(node.id)
        return matched
