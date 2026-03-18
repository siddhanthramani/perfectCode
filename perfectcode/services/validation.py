"""
Validation Service — per-edit checking against the Quality Knowledge Base.

For every significant developer edit, the ValidationService runs a two-phase
pipeline:
  1. **Retrieve** — query all configured IndexerAdapters and merge results.
  2. **Rerank**  — pass merged results through each RerankerAdapter sequentially.
  3. Send the refined results + edit context to an LLM for risk assessment.
  4. Return structured findings to the DevContextAdapter.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from perfectcode.adapters.indexers.base import IndexerAdapter
from perfectcode.adapters.indexers.reranker import RerankerAdapter
from perfectcode.adapters.llm.base import LLMAdapter, LLMCallType, PromptRegistry
from perfectcode.core.models import (
    DevEvent,
    SearchQuery,
    SearchResult,
    Severity,
    TagFilter,
    ValidationFinding,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Orchestrates per-edit validation against the QKB.

    Accepts separate lists of ``IndexerAdapter``s (phase 1: retrieve) and
    ``RerankerAdapter``s (phase 2: refine). Every call to ``validate`` runs
    the full retrieve → rerank → assess cycle.
    """

    def __init__(
        self,
        indexers: list[IndexerAdapter],
        llm: LLMAdapter,
        prompts: PromptRegistry,
        *,
        rerankers: list[RerankerAdapter] | None = None,
        max_results_per_indexer: int = 5,
        min_score: float = 0.3,
    ) -> None:
        self._indexers = indexers
        self._rerankers = rerankers or []
        self._llm = llm
        self._prompts = prompts
        self._max_results = max_results_per_indexer
        self._min_score = min_score

    async def validate(
        self,
        event: DevEvent,
        tag_filter: TagFilter | None = None,
    ) -> ValidationResult:
        similar_issues = await self._search(event, tag_filter)
        if not similar_issues:
            return ValidationResult(dev_event_id=event.id, passed=True, summary="No similar past issues found.")

        findings = await self._assess(event, similar_issues)

        passed = all(f.risk_level in (Severity.LOW, Severity.INFO) for f in findings)
        return ValidationResult(
            dev_event_id=event.id,
            findings=findings,
            passed=passed,
            summary=self._summarize(findings),
        )

    async def _search(
        self,
        event: DevEvent,
        tag_filter: TagFilter | None,
    ) -> list[SearchResult]:
        diff_text = "\n".join(
            f"--- {h.file_path}\n{h.before}\n+++ {h.file_path}\n{h.after}"
            for h in event.diff_hunks
        )
        query = SearchQuery(
            text=event.error_message or diff_text[:500],
            code_snippet=diff_text[:2000],
            language=event.diff_hunks[0].language if event.diff_hunks else None,
            tag_filter=tag_filter,
            max_results=self._max_results,
            min_score=self._min_score,
        )

        # Phase 1: Retrieve from all indexers
        results: list[SearchResult] = []
        for indexer in self._indexers:
            try:
                hits = await indexer.search(query)
                results.extend(hits)
            except Exception:
                logger.exception("Indexer %s failed during retrieval", indexer.indexer_name)

        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for r in results:
            if r.event.id not in seen:
                seen.add(r.event.id)
                deduped.append(r)
        results = deduped

        # Phase 2: Rerank through each reranker sequentially
        for reranker in self._rerankers:
            try:
                results = await reranker.rerank(results, query)
            except Exception:
                logger.exception("Reranker %s failed", reranker.reranker_name)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self._max_results]

    async def _assess(
        self,
        event: DevEvent,
        similar: list[SearchResult],
    ) -> list[ValidationFinding]:
        diff_text = "\n".join(
            f"--- {h.file_path}\n{h.before}\n+++ {h.file_path}\n{h.after}"
            for h in event.diff_hunks
        )
        issues_text = "\n\n".join(
            f"### {r.event.title} (score={r.score:.2f})\n{r.event.description}"
            for r in similar
        )
        prompt = self._prompts.get(LLMCallType.EDIT_VALIDATION)
        response = await self._llm.call(prompt, {"diff": diff_text, "similar_issues": issues_text})

        return self._parse_findings(response.content, similar)

    def _parse_findings(
        self,
        llm_output: str,
        similar: list[SearchResult],
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        try:
            data = json.loads(llm_output)
            items: list[dict[str, Any]] = data if isinstance(data, list) else data.get("findings", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("LLM returned non-JSON for validation; treating as single finding")
            if similar:
                findings.append(
                    ValidationFinding(
                        related_event=similar[0].event,
                        explanation=llm_output[:500],
                    )
                )
            return findings

        events_by_title = {r.event.title: r for r in similar}
        for item in items:
            title = item.get("issue_title", "")
            matched = events_by_title.get(title)
            if not matched and similar:
                matched = similar[0]
            if matched:
                findings.append(
                    ValidationFinding(
                        related_event=matched.event,
                        similarity_score=matched.score,
                        risk_level=Severity(item.get("risk_level", "medium")),
                        explanation=item.get("explanation", ""),
                        suggestion=item.get("suggestion", ""),
                    )
                )
        return findings

    @staticmethod
    def _summarize(findings: list[ValidationFinding]) -> str:
        if not findings:
            return "No issues detected."
        high = sum(1 for f in findings if f.risk_level in (Severity.CRITICAL, Severity.HIGH))
        med = sum(1 for f in findings if f.risk_level == Severity.MEDIUM)
        return f"{len(findings)} potential issue(s): {high} high-risk, {med} medium-risk."
