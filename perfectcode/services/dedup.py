"""
Deduplication Service — detect and cluster near-duplicate QualityEvents.

Runs during ingestion (online) or as a batch job (offline) to prevent the
QKB from accumulating redundant entries.
"""

from __future__ import annotations

import json
import logging

from perfectcode.adapters.indexers.base import IndexerAdapter
from perfectcode.adapters.llm.base import LLMAdapter, LLMCallType, PromptRegistry
from perfectcode.core.models import MetaIssue, QualityEvent

logger = logging.getLogger(__name__)


class DeduplicationService:
    """
    Finds near-duplicates among QualityEvents using an indexer for similarity
    search and an LLM for confirmation.
    """

    def __init__(
        self,
        indexer: IndexerAdapter,
        llm: LLMAdapter,
        prompts: PromptRegistry,
        *,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._indexer = indexer
        self._llm = llm
        self._prompts = prompts
        self._threshold = similarity_threshold
        self._meta_issues: dict[str, MetaIssue] = {}

    async def check_duplicate(self, event: QualityEvent) -> QualityEvent | None:
        """
        Check whether *event* is a near-duplicate of something already indexed.

        Returns the existing duplicate event if found, otherwise None.
        """
        results = await self._indexer.similar(event)
        for result in results:
            if result.score < self._threshold:
                continue
            if await self._confirm_duplicate(event, result.event):
                event.duplicate_of = result.event.id
                logger.info(
                    "Event '%s' marked as duplicate of '%s' (score=%.2f)",
                    event.title,
                    result.event.title,
                    result.score,
                )
                return result.event
        return None

    async def deduplicate_batch(self, events: list[QualityEvent]) -> list[MetaIssue]:
        """
        Run deduplication across a batch of events, clustering duplicates into
        MetaIssues.
        """
        new_meta: list[MetaIssue] = []
        for event in events:
            existing = await self.check_duplicate(event)
            if existing:
                meta = self._get_or_create_meta(existing)
                if event.id not in meta.member_event_ids:
                    meta.member_event_ids.append(event.id)
                if meta not in new_meta:
                    new_meta.append(meta)
        return new_meta

    async def _confirm_duplicate(self, a: QualityEvent, b: QualityEvent) -> bool:
        """Use LLM to confirm whether two events are true duplicates."""
        prompt = self._prompts.get(LLMCallType.SIMILARITY_EXPLAIN)
        response = await self._llm.call(
            prompt,
            {
                "issue_a": f"Title: {a.title}\nDescription: {a.description}",
                "issue_b": f"Title: {b.title}\nDescription: {b.description}",
            },
        )
        try:
            data = json.loads(response.content)
            return bool(data.get("is_duplicate", False))
        except (json.JSONDecodeError, AttributeError):
            return False

    def _get_or_create_meta(self, event: QualityEvent) -> MetaIssue:
        for meta in self._meta_issues.values():
            if event.id in meta.member_event_ids:
                return meta
        meta = MetaIssue(title=event.title, member_event_ids=[event.id])
        self._meta_issues[meta.id] = meta
        return meta
