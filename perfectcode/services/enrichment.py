"""
Enrichment Pipeline — runs CodeEnrichers in a configurable order.

Sits between issue fetching and indexing in the ingestion pipeline.
The user controls which enrichers run, in what order, and the stop condition.
"""

from __future__ import annotations

import logging

from perfectcode.adapters.enrichers.base import (
    CodeEnricher,
    EnrichmentContext,
    EnrichmentResult,
    EnrichmentStrategy,
)
from perfectcode.core.models import QualityEvent

logger = logging.getLogger(__name__)

_SUFFICIENT_THRESHOLD = 0.5


class EnrichmentPipeline:
    """
    Runs a list of ``CodeEnricher``s against each QualityEvent.

    Strategies:
      - ``supplement``: every enricher runs; each adds to what previous ones found.
      - ``first_sufficient``: stop after the first enricher that returns
        confidence >= threshold. Remaining enrichers are skipped.
      - ``all``: every enricher runs unconditionally; results are merged.
        (Same as supplement but ignores ``can_handle``.)
    """

    def __init__(
        self,
        enrichers: list[CodeEnricher],
        strategy: EnrichmentStrategy = EnrichmentStrategy.SUPPLEMENT,
        *,
        sufficient_threshold: float = _SUFFICIENT_THRESHOLD,
    ) -> None:
        self._enrichers = enrichers
        self._strategy = strategy
        self._threshold = sufficient_threshold

    async def enrich(
        self,
        event: QualityEvent,
        context: EnrichmentContext,
    ) -> QualityEvent:
        """Run the enricher pipeline and return the enriched event."""
        accumulated = EnrichmentResult()

        for enricher in self._enrichers:
            if self._strategy != EnrichmentStrategy.ALL and not enricher.can_handle(context):
                logger.debug(
                    "Enricher %s skipped (cannot handle source_type=%s)",
                    enricher.enricher_name,
                    context.source_type,
                )
                continue

            try:
                result = await enricher.enrich(event, context)
                result.enricher_name = enricher.enricher_name
                accumulated = _merge(accumulated, result)

                logger.info(
                    "Enricher %s: found %d locations, %d patches, %d traces (confidence=%.2f)",
                    enricher.enricher_name,
                    len(result.code_locations),
                    len(result.code_patches),
                    len(result.stack_traces),
                    result.confidence,
                )

                if (
                    self._strategy == EnrichmentStrategy.FIRST_SUFFICIENT
                    and accumulated.confidence >= self._threshold
                ):
                    logger.info(
                        "Enrichment sufficient after %s (confidence=%.2f)",
                        enricher.enricher_name,
                        accumulated.confidence,
                    )
                    break

            except Exception:
                logger.exception("Enricher %s failed for event %s", enricher.enricher_name, event.id)

        return _apply(event, accumulated)

    async def enrich_batch(
        self,
        events: list[QualityEvent],
        context: EnrichmentContext,
    ) -> list[QualityEvent]:
        """Enrich every event in the batch."""
        return [await self.enrich(event, context) for event in events]


def _merge(base: EnrichmentResult, new: EnrichmentResult) -> EnrichmentResult:
    """Combine two enrichment results, deduplicating by file path."""
    existing_paths = {loc.file_path for loc in base.code_locations}
    existing_patch_keys = {
        (p.file_path, p.line_start, p.source) for p in base.code_patches
    }

    merged_locations = list(base.code_locations)
    for loc in new.code_locations:
        if loc.file_path not in existing_paths:
            merged_locations.append(loc)
            existing_paths.add(loc.file_path)

    merged_patches = list(base.code_patches)
    for patch in new.code_patches:
        key = (patch.file_path, patch.line_start, patch.source)
        if key not in existing_patch_keys:
            merged_patches.append(patch)
            existing_patch_keys.add(key)

    existing_traces = set(base.stack_traces)
    merged_traces = list(base.stack_traces)
    for trace in new.stack_traces:
        if trace not in existing_traces:
            merged_traces.append(trace)
            existing_traces.add(trace)

    return EnrichmentResult(
        code_locations=merged_locations,
        code_patches=merged_patches,
        stack_traces=merged_traces,
        confidence=max(base.confidence, new.confidence),
    )


def _apply(event: QualityEvent, result: EnrichmentResult) -> QualityEvent:
    """Write enrichment results back onto the QualityEvent."""
    existing_loc_paths = {loc.file_path for loc in event.code_locations}
    for loc in result.code_locations:
        if loc.file_path not in existing_loc_paths:
            event.code_locations.append(loc)
            existing_loc_paths.add(loc.file_path)

    existing_patch_keys = {
        (p.file_path, p.line_start, p.source) for p in event.code_patches
    }
    for patch in result.code_patches:
        key = (patch.file_path, patch.line_start, patch.source)
        if key not in existing_patch_keys:
            event.code_patches.append(patch)
            existing_patch_keys.add(key)

    existing_traces = set(event.stack_traces)
    for trace in result.stack_traces:
        if trace not in existing_traces:
            event.stack_traces.append(trace)

    return event
