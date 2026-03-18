"""
Ingestion Orchestrator — schedules and runs issue ingestion pipelines.

The orchestrator pulls issues from configured IssueSourceAdapters, normalizes
them into QualityEvents, enriches them with code data (via the EnrichmentPipeline),
tags them with the taxonomy, indexes them in all configured IndexerAdapters,
and optionally runs deduplication.

Pipeline order:  fetch → enrich → tag → index → deduplicate
"""

from __future__ import annotations

import logging
from datetime import datetime

from perfectcode.adapters.enrichers.base import EnrichmentContext
from perfectcode.adapters.indexers.base import IndexerAdapter
from perfectcode.adapters.issues.base import IssueSourceAdapter
from perfectcode.adapters.llm.base import LLMAdapter, PromptRegistry
from perfectcode.core.models import QualityEvent
from perfectcode.core.taxonomy import QualityTaxonomyEngine
from perfectcode.services.dedup import DeduplicationService
from perfectcode.services.enrichment import EnrichmentPipeline

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """
    Central pipeline that turns external issues into indexed, tagged QualityEvents.

    Usage:
        orchestrator = IngestionOrchestrator(sources, indexers, taxonomy_engine, llm, prompts)
        await orchestrator.run_incremental()   # daily
        await orchestrator.run_full_backfill() # one-time
    """

    def __init__(
        self,
        sources: list[IssueSourceAdapter],
        indexers: list[IndexerAdapter],
        taxonomy: QualityTaxonomyEngine,
        llm: LLMAdapter,
        prompts: PromptRegistry,
        dedup: DeduplicationService | None = None,
        enrichment: EnrichmentPipeline | None = None,
        enrichment_contexts: dict[str, EnrichmentContext] | None = None,
    ) -> None:
        self._sources = sources
        self._indexers = indexers
        self._taxonomy = taxonomy
        self._llm = llm
        self._prompts = prompts
        self._dedup = dedup
        self._enrichment = enrichment
        self._enrichment_contexts = enrichment_contexts or {}

    async def run_incremental(self, since: datetime | None = None) -> list[QualityEvent]:
        """Pull new/updated issues since *since* (default: last 24h) and ingest."""
        if since is None:
            from datetime import timedelta

            since = datetime.utcnow() - timedelta(days=1)

        all_events: list[QualityEvent] = []
        for source in self._sources:
            try:
                events = await source.fetch_updated_since(since)
                logger.info("Fetched %d events from %s", len(events), source.source_name)
                all_events.extend(events)
            except Exception:
                logger.exception("Failed to fetch from %s", source.source_name)

        return await self._ingest(all_events)

    async def run_full_backfill(self) -> list[QualityEvent]:
        """Pull all issues from every source and ingest."""
        all_events: list[QualityEvent] = []
        for source in self._sources:
            try:
                events = await source.fetch_all()
                logger.info("Backfill: fetched %d events from %s", len(events), source.source_name)
                all_events.extend(events)
            except Exception:
                logger.exception("Backfill failed for %s", source.source_name)

        return await self._ingest(all_events)

    async def _ingest(self, events: list[QualityEvent]) -> list[QualityEvent]:
        """Enrich → tag → index → deduplicate."""
        if not events:
            return []

        if self._enrichment:
            for event in events:
                source_type = event.source.value
                context = self._enrichment_contexts.get(
                    source_type,
                    EnrichmentContext(source_type=source_type),
                )
                try:
                    await self._enrichment.enrich(event, context)
                except Exception:
                    logger.exception("Enrichment failed for event %s", event.id)

        for event in events:
            try:
                await self._taxonomy.auto_tag(event, self._llm, self._prompts)
            except Exception:
                logger.exception("Taxonomy tagging failed for event %s", event.id)

        for indexer in self._indexers:
            try:
                await indexer.index(events)
                logger.info("Indexed %d events in %s", len(events), indexer.indexer_name)
            except Exception:
                logger.exception("Indexing failed for %s", indexer.indexer_name)

        if self._dedup:
            try:
                await self._dedup.deduplicate_batch(events)
            except Exception:
                logger.exception("Deduplication failed")

        logger.info("Ingestion complete: %d events processed", len(events))
        return events
