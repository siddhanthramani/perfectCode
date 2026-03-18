"""
LLM-based Code Enricher — uses an LLM to extract structured code data.

Sends the issue description to an LLM and asks it to identify:
  - File paths and line numbers mentioned in the text
  - Stack traces
  - Code snippets and their context

This is the fallback enricher that catches anything regex missed.
Source-agnostic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from perfectcode.adapters.enrichers.base import (
    CodeEnricher,
    EnrichmentContext,
    EnrichmentResult,
)
from perfectcode.adapters.llm.base import LLMAdapter, PromptRegistry
from perfectcode.core.models import (
    CodeLocation,
    CodePatch,
    LLMResponse,
    PromptTemplate,
    QualityEvent,
)

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = PromptTemplate(
    name="code_extraction",
    description="Extract structured code data from an issue description",
    system_prompt=(
        "You are a code analysis expert. Given a bug report or issue description, "
        "extract all code-related information you can find."
    ),
    user_template=(
        "## Issue\nTitle: {title}\n\n## Description\n{description}\n\n"
        "Extract the following as JSON:\n"
        '- "file_paths": list of objects with "path", "line_start" (int or null), "function_name" (str or null)\n'
        '- "stack_traces": list of stack trace strings found in the text\n'
        '- "code_snippets": list of objects with "file_path", "code", "language" (str or null), '
        '"is_buggy" (bool — true if this is the problematic code, false if it\'s the fix)\n'
        "\nReturn ONLY valid JSON. If nothing is found, return empty lists."
    ),
    temperature=0.1,
    max_tokens=2048,
)


class LLMEnricher(CodeEnricher):
    """
    Uses an LLM to extract code locations, snippets, and stack traces
    from issue descriptions. Source-agnostic fallback enricher.
    """

    def __init__(self, llm: LLMAdapter, prompts: PromptRegistry | None = None) -> None:
        self._llm = llm
        self._prompts = prompts
        if prompts:
            prompts.register("code_extraction", _EXTRACTION_PROMPT)

    @property
    def enricher_name(self) -> str:
        return "llm_extraction"

    def can_handle(self, context: EnrichmentContext) -> bool:
        return True

    async def enrich(
        self,
        event: QualityEvent,
        context: EnrichmentContext,
    ) -> EnrichmentResult:
        if not event.description and not event.title:
            return EnrichmentResult(enricher_name=self.enricher_name)

        prompt = _EXTRACTION_PROMPT
        if self._prompts:
            try:
                prompt = self._prompts.get("code_extraction")
            except KeyError:
                pass

        try:
            response = await self._llm.call(
                prompt,
                {"title": event.title, "description": event.description},
            )
            return self._parse_response(response)
        except Exception:
            logger.exception("LLM enrichment failed for event %s", event.id)
            return EnrichmentResult(enricher_name=self.enricher_name)

    def _parse_response(self, response: LLMResponse) -> EnrichmentResult:
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for code extraction")
            return EnrichmentResult(enricher_name=self.enricher_name)

        locations: list[CodeLocation] = []
        for fp in data.get("file_paths", []):
            if isinstance(fp, dict) and fp.get("path"):
                locations.append(CodeLocation(
                    file_path=fp["path"],
                    line_start=fp.get("line_start"),
                    function_name=fp.get("function_name"),
                ))

        stack_traces: list[str] = [
            t for t in data.get("stack_traces", []) if isinstance(t, str) and t.strip()
        ]

        patches: list[CodePatch] = []
        for snippet in data.get("code_snippets", []):
            if not isinstance(snippet, dict):
                continue
            code = snippet.get("code", "")
            if not code:
                continue
            is_buggy = snippet.get("is_buggy", True)
            patches.append(CodePatch(
                file_path=snippet.get("file_path", "(llm extracted)"),
                language=snippet.get("language"),
                before=code if is_buggy else "",
                after="" if is_buggy else code,
                source="llm_extraction",
            ))

        has_data = bool(locations or patches or stack_traces)
        confidence = 0.4 if has_data else 0.0
        if len(patches) >= 2:
            confidence = 0.6
        if locations and patches:
            confidence = 0.7

        return EnrichmentResult(
            code_locations=locations,
            code_patches=patches,
            stack_traces=stack_traces,
            confidence=min(confidence, 1.0),
            enricher_name=self.enricher_name,
        )
