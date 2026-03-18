"""
Smart mock LLM adapter for testing and development.

Returns plausible, keyword-driven JSON responses so the full pipeline
(tagging, validation, deduplication) produces meaningful output without
calling a real model. Fully offline, zero API keys.
"""

from __future__ import annotations

import json
import re

from perfectcode.adapters.llm.base import LLMAdapter, LLMCallType
from perfectcode.core.models import LLMResponse, PromptTemplate

_TAG_RULES: list[tuple[list[str], str, str, str]] = [
    (
        ["none", "null", "nonetype", "attributeerror", "optional", "undefined"],
        "NullAccess",
        "Bugs/Language",
        "Accessing attributes on potentially null/None values",
    ),
    (
        ["sql", "injection", "query", "unsanitized", "interpolat", "raw sql"],
        "SQLInjection",
        "Bugs/Security",
        "SQL injection via unsanitized input",
    ),
    (
        ["xss", "script", "html", "sanitize", "escape"],
        "XSS",
        "Bugs/Security",
        "Cross-site scripting vulnerabilities",
    ),
    (
        ["off-by-one", "index", "boundary", "range", "overflow", "underflow", "bounds"],
        "BoundaryError",
        "Bugs/Logic",
        "Off-by-one and boundary condition errors",
    ),
    (
        ["race", "concurrent", "thread", "lock", "deadlock", "async", "await"],
        "Concurrency",
        "Bugs/Runtime",
        "Race conditions and concurrency issues",
    ),
    (
        ["memory", "leak", "oom", "gc", "garbage"],
        "MemoryLeak",
        "Bugs/Runtime",
        "Memory leaks and excessive allocation",
    ),
    (
        ["type", "typeerror", "cast", "conversion", "mismatch", "int", "str", "float"],
        "TypeError",
        "Bugs/Language",
        "Type mismatches and invalid conversions",
    ),
    (
        ["import", "module", "dependency", "package", "require"],
        "ImportError",
        "Bugs/Language",
        "Missing or broken imports and dependencies",
    ),
    (
        ["ui", "css", "layout", "render", "display", "style", "frontend", "visual"],
        "UIDefect",
        "Bugs/UI",
        "Visual and layout defects",
    ),
    (
        ["api", "endpoint", "request", "response", "http", "status", "rest"],
        "APIError",
        "Bugs/API",
        "API endpoint failures and incorrect responses",
    ),
    (
        ["auth", "permission", "access", "denied", "forbidden", "token", "login"],
        "AuthError",
        "Bugs/Security",
        "Authentication and authorization failures",
    ),
    (
        ["config", "setting", "environment", "env", "variable"],
        "ConfigError",
        "Bugs/Config",
        "Configuration and environment issues",
    ),
    (
        ["test", "assert", "fail", "expect", "fixture"],
        "TestFailure",
        "Bugs/Testing",
        "Test failures and broken assertions",
    ),
    (
        ["performance", "slow", "timeout", "latency", "bottleneck"],
        "Performance",
        "Bugs/Performance",
        "Performance degradation and timeouts",
    ),
]


def _match_tags(text: str) -> list[dict[str, str]]:
    """Match text against keyword rules and return tag suggestions."""
    lower = text.lower()
    matched: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for keywords, name, parent_path, description in _TAG_RULES:
        if any(kw in lower for kw in keywords):
            if name not in seen_names:
                seen_names.add(name)
                parts = parent_path.split("/")
                matched.append({
                    "name": name,
                    "parent_path": parent_path,
                    "description": description,
                    "category": parts[-1] if parts else "General",
                })
    if not matched:
        matched.append({
            "name": "General",
            "parent_path": "Bugs",
            "description": "General quality issue",
            "category": "General",
        })
    return matched


def _extract_findings(diff: str, issues_text: str) -> list[dict[str, str]]:
    """Compare diff keywords against retrieved issues to generate findings."""
    diff_lower = diff.lower()
    findings: list[dict[str, str]] = []

    issue_blocks = re.split(r"###\s+", issues_text)
    for block in issue_blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        title_line = lines[0] if lines else ""
        title_match = re.match(r"(.+?)\s*\(score=", title_line)
        title = title_match.group(1).strip() if title_match else title_line.strip()
        body = "\n".join(lines[1:]).lower()

        overlap_words = set(body.split()) & set(diff_lower.split())
        meaningful = {w for w in overlap_words if len(w) > 3}

        if len(meaningful) >= 2 or any(kw in diff_lower for kw in ["none", "null", "sql", "inject", "index", "range"]):
            risk = "high" if any(kw in diff_lower for kw in ["sql", "inject", "none", "null"]) else "medium"
            findings.append({
                "issue_title": title,
                "risk_level": risk,
                "explanation": f"This edit shares patterns with a known issue: {title}. "
                               f"Overlapping concepts: {', '.join(list(meaningful)[:5]) or 'structural similarity'}.",
                "suggestion": f"Review the fix for '{title}' and ensure this edit does not reintroduce the same problem.",
            })

    return findings


class MockLLMAdapter(LLMAdapter):
    """
    Keyword-driven mock that produces meaningful output for demos.

    Uses simple keyword matching to suggest tags, detect validation findings,
    and assess similarity — enough to demonstrate the full pipeline without
    any API keys.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    async def call(self, prompt: PromptTemplate, variables: dict[str, str]) -> LLMResponse:
        if prompt.name == LLMCallType.TAG_SUGGESTION.value:
            content = self._tag_suggestion(variables)
        elif prompt.name == LLMCallType.EDIT_VALIDATION.value:
            content = self._edit_validation(variables)
        elif prompt.name == LLMCallType.SIMILARITY_EXPLAIN.value:
            content = self._similarity_explain(variables)
        elif prompt.name == LLMCallType.TAXONOMY_REORG.value:
            content = json.dumps([])
        else:
            content = json.dumps({"result": "mock response"})

        return LLMResponse(content=content, model="mock-smart")

    async def health_check(self) -> bool:
        return True

    @staticmethod
    def _tag_suggestion(variables: dict[str, str]) -> str:
        text = f"{variables.get('title', '')} {variables.get('description', '')}"
        tags = _match_tags(text)

        taxonomy_excerpt = variables.get("taxonomy_excerpt", "")
        existing_ids: list[str] = []
        new_tags: list[dict[str, str]] = []

        for tag in tags:
            tag_id_match = re.search(
                rf"\[([a-f0-9]+)\]\s+{re.escape(tag['name'])}",
                taxonomy_excerpt,
            )
            if tag_id_match:
                existing_ids.append(tag_id_match.group(1))
            else:
                parent_match = re.search(
                    rf"\[([a-f0-9]+)\]\s+{re.escape(tag['category'])}",
                    taxonomy_excerpt,
                )
                parent_id = parent_match.group(1) if parent_match else None
                new_tags.append({
                    "name": tag["name"],
                    "parent_id": parent_id,
                    "description": tag["description"],
                })

        return json.dumps({"existing_tag_ids": existing_ids, "new_tags": new_tags})

    @staticmethod
    def _edit_validation(variables: dict[str, str]) -> str:
        diff = variables.get("diff", "")
        issues = variables.get("similar_issues", "")
        findings = _extract_findings(diff, issues)
        return json.dumps({"findings": findings})

    @staticmethod
    def _similarity_explain(variables: dict[str, str]) -> str:
        a = variables.get("issue_a", "").lower()
        b = variables.get("issue_b", "").lower()
        words_a = set(a.split())
        words_b = set(b.split())
        overlap = words_a & words_b
        meaningful = {w for w in overlap if len(w) > 3}
        ratio = len(meaningful) / max(len(words_a | words_b), 1)

        return json.dumps({
            "is_duplicate": ratio > 0.4,
            "confidence": round(min(ratio * 2, 1.0), 2),
            "explanation": f"Shared {len(meaningful)} significant terms. "
                           f"{'Likely the same issue.' if ratio > 0.4 else 'Different issues with some overlap.'}",
        })
