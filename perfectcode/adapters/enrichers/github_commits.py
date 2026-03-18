"""
GitHub Commit Search Enricher.

Searches for commits that reference an issue number (e.g. "fixes #123",
"closes #456") and extracts file diffs from those commits. Works as a
fallback when no linked PR is found.
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
from perfectcode.adapters.enrichers.github_pr import _guess_language, _parse_unified_diff
from perfectcode.core.models import CodeLocation, CodePatch, QualityEvent

logger = logging.getLogger(__name__)


class GitHubCommitSearchEnricher(CodeEnricher):
    """
    Searches commits mentioning the issue number and extracts diffs.
    """

    @property
    def enricher_name(self) -> str:
        return "github_commit_search"

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
            commit_shas = await self._search_commits(client, owner, repo, issue_number)
            if not commit_shas:
                return EnrichmentResult(enricher_name=self.enricher_name)

            locations: list[CodeLocation] = []
            patches: list[CodePatch] = []

            for sha in commit_shas[:3]:
                files = await self._get_commit_files(client, owner, repo, sha)
                for file_info in files:
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
                        source="github_commit",
                    ))

            confidence = min(len(patches) / 3, 1.0) if patches else 0.0
            return EnrichmentResult(
                code_locations=locations,
                code_patches=patches,
                confidence=confidence,
                enricher_name=self.enricher_name,
            )

    async def _search_commits(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        issue_number: str,
    ) -> list[str]:
        """Search for commits that reference the issue number."""
        keywords = ["fixes", "closes", "resolves", "fix", "close", "resolve"]
        query_parts = [f"repo:{owner}/{repo}"]
        query_parts.append(f"#{issue_number}")

        try:
            resp = await client.get(
                "/search/commits",
                params={
                    "q": " ".join(query_parts),
                    "per_page": 5,
                    "sort": "committer-date",
                    "order": "desc",
                },
                headers={"Accept": "application/vnd.github.cloak-preview+json"},
            )
            if resp.status_code != 200:
                return []
            return [item["sha"] for item in resp.json().get("items", [])]
        except Exception:
            logger.debug("Commit search failed for %s/%s#%s", owner, repo, issue_number)
            return []

    async def _get_commit_files(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        sha: str,
    ) -> list[dict]:
        """Get changed files for a single commit."""
        try:
            resp = await client.get(f"/repos/{owner}/{repo}/commits/{sha}")
            if resp.status_code != 200:
                return []
            return resp.json().get("files", [])
        except Exception:
            return []
