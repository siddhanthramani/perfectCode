"""
GitHub Linked PR Enricher.

Follows the chain: issue → linked pull requests → file diffs.
Extracts actual before/after code patches from the PR that fixed the bug.
This is the highest-value enricher — it gives you the exact code pattern
that caused the bug and how it was fixed.
"""

from __future__ import annotations

import logging
import re

import httpx

from perfectcode.adapters.enrichers.base import (
    CodeEnricher,
    EnrichmentContext,
    EnrichmentResult,
)
from perfectcode.core.models import CodeLocation, CodePatch, QualityEvent

logger = logging.getLogger(__name__)

_LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".cpp": "cpp", ".c": "c", ".cs": "csharp", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
}


def _guess_language(file_path: str) -> str | None:
    for ext, lang in _LANG_MAP.items():
        if file_path.endswith(ext):
            return lang
    return None


class GitHubLinkedPREnricher(CodeEnricher):
    """
    Finds PRs that reference a GitHub issue, fetches their file diffs,
    and extracts CodePatches with before/after code.
    """

    @property
    def enricher_name(self) -> str:
        return "github_linked_pr"

    def can_handle(self, context: EnrichmentContext) -> bool:
        return context.source_type == "github"

    async def enrich(
        self,
        event: QualityEvent,
        context: EnrichmentContext,
    ) -> EnrichmentResult:
        if not event.external_id:
            return EnrichmentResult(enricher_name=self.enricher_name)

        issue_number = event.external_id
        owner = context.owner
        repo = context.repo
        headers = {"Accept": "application/vnd.github+json", **context.auth_headers}

        async with httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
        ) as client:
            pr_numbers = await self._find_linked_prs(client, owner, repo, issue_number)
            if not pr_numbers:
                return EnrichmentResult(enricher_name=self.enricher_name)

            locations: list[CodeLocation] = []
            patches: list[CodePatch] = []

            for pr_num in pr_numbers[:3]:
                pr_patches = await self._get_pr_files(client, owner, repo, pr_num)
                for file_info in pr_patches:
                    filename = file_info.get("filename", "")
                    patch_text = file_info.get("patch", "")
                    if not filename or not patch_text:
                        continue

                    before, after, line_start = _parse_unified_diff(patch_text)
                    locations.append(CodeLocation(
                        file_path=filename,
                        line_start=line_start,
                        repo=f"{owner}/{repo}",
                    ))
                    patches.append(CodePatch(
                        file_path=filename,
                        language=_guess_language(filename),
                        before=before,
                        after=after,
                        line_start=line_start,
                        source="github_pr",
                    ))

            confidence = min(len(patches) / 3, 1.0) if patches else 0.0
            return EnrichmentResult(
                code_locations=locations,
                code_patches=patches,
                confidence=confidence,
                enricher_name=self.enricher_name,
            )

    async def _find_linked_prs(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        issue_number: str,
    ) -> list[int]:
        """Find PRs linked to this issue via timeline events."""
        pr_numbers: list[int] = []
        try:
            resp = await client.get(
                f"/repos/{owner}/{repo}/issues/{issue_number}/timeline",
                headers={"Accept": "application/vnd.github.mockingbird-preview+json"},
            )
            if resp.status_code != 200:
                return pr_numbers

            for event_data in resp.json():
                if event_data.get("event") == "cross-referenced":
                    source = event_data.get("source", {}).get("issue", {})
                    if source.get("pull_request"):
                        pr_numbers.append(source["number"])
                elif event_data.get("event") == "connected":
                    pass
        except Exception:
            logger.debug("Timeline API failed for %s/%s#%s", owner, repo, issue_number)

        if not pr_numbers:
            pr_numbers = await self._search_referencing_prs(client, owner, repo, issue_number)

        return pr_numbers

    async def _search_referencing_prs(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        issue_number: str,
    ) -> list[int]:
        """Fallback: search for PRs that mention the issue number."""
        try:
            resp = await client.get(
                "/search/issues",
                params={
                    "q": f"repo:{owner}/{repo} is:pr #{issue_number}",
                    "per_page": 5,
                },
            )
            if resp.status_code != 200:
                return []
            return [item["number"] for item in resp.json().get("items", [])]
        except Exception:
            return []

    async def _get_pr_files(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> list[dict]:
        """Get the list of changed files in a PR."""
        try:
            resp = await client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
                params={"per_page": 30},
            )
            if resp.status_code != 200:
                return []
            return resp.json()
        except Exception:
            return []


def _parse_unified_diff(patch: str) -> tuple[str, str, int | None]:
    """
    Parse a unified diff patch into before/after content and starting line.
    Returns (before_lines, after_lines, line_start).
    """
    before_lines: list[str] = []
    after_lines: list[str] = []
    line_start: int | None = None

    for line in patch.split("\n"):
        hunk_match = re.match(r"^@@ -(\d+)", line)
        if hunk_match and line_start is None:
            line_start = int(hunk_match.group(1))
            continue

        if line.startswith("-") and not line.startswith("---"):
            before_lines.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            after_lines.append(line[1:])
        elif not line.startswith("@@"):
            before_lines.append(line)
            after_lines.append(line)

    return "\n".join(before_lines), "\n".join(after_lines), line_start
