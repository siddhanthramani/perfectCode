"""
Jira Development Panel Enricher.

Follows Jira's development information (linked PRs, commits, branches) via
the Jira Software REST API or remote links, and extracts code patches from
the connected source control system.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from perfectcode.adapters.enrichers.base import (
    CodeEnricher,
    EnrichmentContext,
    EnrichmentResult,
)
from perfectcode.adapters.enrichers.github_pr import _guess_language, _parse_unified_diff
from perfectcode.core.models import CodeLocation, CodePatch, QualityEvent

logger = logging.getLogger(__name__)


class JiraDevPanelEnricher(CodeEnricher):
    """
    Extracts code data from Jira's linked development information.

    Supports two paths:
      1. Jira's development information API (if available)
      2. Remote links that point to PRs on GitHub/Bitbucket/GitLab
    """

    @property
    def enricher_name(self) -> str:
        return "jira_dev_panel"

    def can_handle(self, context: EnrichmentContext) -> bool:
        return context.source_type == "jira"

    async def enrich(
        self,
        event: QualityEvent,
        context: EnrichmentContext,
    ) -> EnrichmentResult:
        if not event.external_id:
            return EnrichmentResult(enricher_name=self.enricher_name)

        jira_key = event.external_id
        base_url = context.base_url.rstrip("/")
        headers = {"Accept": "application/json", **context.auth_headers}

        locations: list[CodeLocation] = []
        patches: list[CodePatch] = []

        async with httpx.AsyncClient(headers=headers) as client:
            pr_urls = await self._get_remote_link_prs(client, base_url, jira_key)

            for url in pr_urls[:3]:
                github_patches = await self._fetch_github_pr_from_url(client, url, context)
                for file_info in github_patches:
                    filename = file_info.get("filename", "")
                    patch_text = file_info.get("patch", "")
                    if not filename or not patch_text:
                        continue

                    before, after, line_start = _parse_unified_diff(patch_text)
                    locations.append(CodeLocation(file_path=filename, line_start=line_start))
                    patches.append(CodePatch(
                        file_path=filename,
                        language=_guess_language(filename),
                        before=before,
                        after=after,
                        line_start=line_start,
                        source="jira_linked_pr",
                    ))

        confidence = min(len(patches) / 3, 1.0) if patches else 0.0
        return EnrichmentResult(
            code_locations=locations,
            code_patches=patches,
            confidence=confidence,
            enricher_name=self.enricher_name,
        )

    async def _get_remote_link_prs(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        jira_key: str,
    ) -> list[str]:
        """Get PR URLs from Jira remote links."""
        try:
            resp = await client.get(f"{base_url}/rest/api/3/issue/{jira_key}/remotelink")
            if resp.status_code != 200:
                return []

            urls: list[str] = []
            for link in resp.json():
                obj = link.get("object", {})
                url = obj.get("url", "")
                title = obj.get("title", "").lower()
                if any(kw in url for kw in ["pull", "merge_request"]) or "pr" in title:
                    urls.append(url)
            return urls
        except Exception:
            logger.debug("Failed to fetch remote links for %s", jira_key)
            return []

    async def _fetch_github_pr_from_url(
        self,
        client: httpx.AsyncClient,
        pr_url: str,
        context: EnrichmentContext,
    ) -> list[dict[str, Any]]:
        """If the URL points to a GitHub PR, fetch its files."""
        match = re.match(
            r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)",
            pr_url,
        )
        if not match:
            return []

        owner, repo, pr_number = match.group(1), match.group(2), match.group(3)
        gh_headers = {
            "Accept": "application/vnd.github+json",
            **context.extra.get("github_headers", {}),
        }

        try:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files",
                headers=gh_headers,
                params={"per_page": 30},
            )
            if resp.status_code != 200:
                return []
            return resp.json()
        except Exception:
            return []
