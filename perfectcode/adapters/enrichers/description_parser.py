"""
Description Parser Enricher — source-agnostic regex extraction.

Parses the issue description text to extract:
  - Stack traces (Python, Java, JavaScript, Go, etc.)
  - Code blocks (fenced with ```)
  - File path references (e.g. ``src/api/views.py:42``)

Works with any issue source since it only looks at the description text.
"""

from __future__ import annotations

import re

from perfectcode.adapters.enrichers.base import (
    CodeEnricher,
    EnrichmentContext,
    EnrichmentResult,
)
from perfectcode.core.models import CodeLocation, CodePatch, QualityEvent

_PYTHON_TRACEBACK = re.compile(
    r'(Traceback \(most recent call last\):.*?(?:\w+Error|\w+Exception): .+)',
    re.DOTALL,
)

_JAVA_STACKTRACE = re.compile(
    r'((?:\w+\.)+\w+(?:Error|Exception): .+?(?:\n\s+at .+)+)',
    re.DOTALL,
)

_JS_STACKTRACE = re.compile(
    r'((?:TypeError|ReferenceError|SyntaxError|RangeError|Error): .+?(?:\n\s+at .+)+)',
    re.DOTALL,
)

_GO_PANIC = re.compile(
    r'(goroutine \d+ \[.+?\]:.*?(?:\n\t.+)+)',
    re.DOTALL,
)

_CODE_FENCE = re.compile(
    r'```(\w*)\n(.*?)```',
    re.DOTALL,
)

_FILE_PATH_PATTERN = re.compile(
    r'(?:^|\s|["\'])'
    r'((?:[\w./\\-]+/)?[\w.-]+\.(?:py|js|ts|jsx|tsx|java|go|rs|rb|cpp|c|cs|php|swift|kt|scala))'
    r'(?::(\d+))?'
    r'(?:\s|["\']|$|,|\))',
)

_PYTHON_FILE_LINE = re.compile(
    r'File "([^"]+)", line (\d+)',
)


class DescriptionParserEnricher(CodeEnricher):
    """
    Regex-based extraction of stack traces, code blocks, and file paths
    from issue descriptions. Source-agnostic — works with GitHub, Jira, or
    any other tracker.
    """

    @property
    def enricher_name(self) -> str:
        return "description_parser"

    def can_handle(self, context: EnrichmentContext) -> bool:
        return True

    async def enrich(
        self,
        event: QualityEvent,
        context: EnrichmentContext,
    ) -> EnrichmentResult:
        text = f"{event.title}\n{event.description}"
        if not text.strip():
            return EnrichmentResult(enricher_name=self.enricher_name)

        stack_traces = self._extract_stack_traces(text)
        code_patches = self._extract_code_blocks(text)
        code_locations = self._extract_file_paths(text)

        for trace in stack_traces:
            trace_locations = self._extract_file_paths(trace)
            for loc in trace_locations:
                if not any(l.file_path == loc.file_path and l.line_start == loc.line_start for l in code_locations):
                    code_locations.append(loc)

        has_data = bool(stack_traces or code_patches or code_locations)
        confidence = 0.0
        if has_data:
            confidence = 0.3
            if code_patches:
                confidence += 0.2
            if stack_traces:
                confidence += 0.2
            if len(code_locations) >= 2:
                confidence += 0.1

        return EnrichmentResult(
            code_locations=code_locations,
            code_patches=code_patches,
            stack_traces=stack_traces,
            confidence=min(confidence, 1.0),
            enricher_name=self.enricher_name,
        )

    def _extract_stack_traces(self, text: str) -> list[str]:
        traces: list[str] = []
        for pattern in [_PYTHON_TRACEBACK, _JAVA_STACKTRACE, _JS_STACKTRACE, _GO_PANIC]:
            for match in pattern.finditer(text):
                trace = match.group(1).strip()
                if trace and trace not in traces:
                    traces.append(trace)
        return traces

    def _extract_code_blocks(self, text: str) -> list[CodePatch]:
        patches: list[CodePatch] = []
        for match in _CODE_FENCE.finditer(text):
            lang = match.group(1) or None
            code = match.group(2).strip()
            if not code or len(code) < 5:
                continue
            patches.append(CodePatch(
                file_path="(from description)",
                language=lang,
                before=code,
                source="description",
            ))
        return patches

    def _extract_file_paths(self, text: str) -> list[CodeLocation]:
        locations: list[CodeLocation] = []
        seen: set[tuple[str, int | None]] = set()

        for match in _PYTHON_FILE_LINE.finditer(text):
            path = match.group(1)
            line = int(match.group(2))
            key = (path, line)
            if key not in seen:
                seen.add(key)
                locations.append(CodeLocation(file_path=path, line_start=line))

        for match in _FILE_PATH_PATTERN.finditer(text):
            path = match.group(1)
            line = int(match.group(2)) if match.group(2) else None
            key = (path, line)
            if key not in seen:
                seen.add(key)
                locations.append(CodeLocation(file_path=path, line_start=line))

        return locations
