"""
GitHub Issues source adapter (skeleton).

Pulls issues labeled as bugs from a GitHub repository and normalizes them
into QualityEvents.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from perfectcode.adapters.issues.base import IssueSourceAdapter
from perfectcode.core.models import (
    EventSource,
    EventStatus,
    QualityEvent,
    Severity,
)


class GitHubIssueSourceAdapter(IssueSourceAdapter):
    """
    Pull issues from a GitHub repo via the REST API.

    Args:
      - owner / repo: e.g. ``"myorg"`` / ``"myrepo"``
      - token: optional GitHub personal access token (60 req/hr without,
        5000 req/hr with a free token)
      - bug_label: label used to identify bug issues (default ``"bug"``)
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str | None = None,
        *,
        bug_label: str = "bug",
        max_issues: int = 0,
    ) -> None:
        self._owner = owner
        self._repo = repo
        self._bug_label = bug_label
        self._max_issues = max_issues
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
        )

    @property
    def source_name(self) -> str:
        return "github"

    async def fetch_updated_since(self, since: datetime) -> list[QualityEvent]:
        params: dict[str, Any] = {
            "labels": self._bug_label,
            "since": since.isoformat(),
            "state": "all",
            "per_page": 100,
            "sort": "updated",
            "direction": "desc",
        }
        return await self._paginate(params)

    async def fetch_all(self) -> list[QualityEvent]:
        params: dict[str, Any] = {
            "labels": self._bug_label,
            "state": "all",
            "per_page": 100,
            "sort": "created",
            "direction": "desc",
        }
        return await self._paginate(params)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"/repos/{self._owner}/{self._repo}")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _paginate(self, params: dict[str, Any]) -> list[QualityEvent]:
        events: list[QualityEvent] = []
        page = 1
        while True:
            params["page"] = page
            resp = await self._client.get(
                f"/repos/{self._owner}/{self._repo}/issues",
                params=params,
            )
            resp.raise_for_status()
            issues = resp.json()
            if not issues:
                break
            for issue in issues:
                if "pull_request" in issue:
                    continue
                events.append(self._normalize(issue))
                if self._max_issues and len(events) >= self._max_issues:
                    return events
            page += 1
        return events

    def _normalize(self, issue: dict[str, Any]) -> QualityEvent:
        state = issue.get("state", "open")
        status = EventStatus.CLOSED if state == "closed" else EventStatus.OPEN

        labels = [lbl["name"] for lbl in issue.get("labels", []) if isinstance(lbl, dict)]
        severity = Severity.MEDIUM
        for label in labels:
            lower = label.lower()
            if "critical" in lower:
                severity = Severity.CRITICAL
            elif "high" in lower or "major" in lower:
                severity = Severity.HIGH
            elif "low" in lower or "minor" in lower:
                severity = Severity.LOW

        return QualityEvent(
            source=EventSource.GITHUB,
            external_id=str(issue["number"]),
            title=issue.get("title", ""),
            description=issue.get("body", "") or "",
            status=status,
            severity=severity,
            metadata={
                "url": issue.get("html_url", ""),
                "labels": labels,
                "author": issue.get("user", {}).get("login", ""),
            },
            created_at=datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
            if issue.get("created_at")
            else datetime.utcnow(),
        )
