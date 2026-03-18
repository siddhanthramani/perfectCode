"""
Jira issue source adapter (skeleton).

Connects to a Jira instance via REST API and normalizes Jira issues into
QualityEvents. Requires ``httpx`` and Jira API credentials.
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


_STATUS_MAP: dict[str, EventStatus] = {
    "To Do": EventStatus.OPEN,
    "In Progress": EventStatus.IN_PROGRESS,
    "Done": EventStatus.RESOLVED,
    "Closed": EventStatus.CLOSED,
}

_PRIORITY_MAP: dict[str, Severity] = {
    "Blocker": Severity.CRITICAL,
    "Critical": Severity.CRITICAL,
    "Major": Severity.HIGH,
    "Minor": Severity.MEDIUM,
    "Trivial": Severity.LOW,
}


class JiraIssueSourceAdapter(IssueSourceAdapter):
    """
    Pull issues from Jira Cloud or Server via REST API.

    Requires:
      - base_url: e.g. ``https://yourorg.atlassian.net``
      - email + api_token (Cloud) or personal access token (Server)
      - project_key: Jira project to query (e.g. ``"PROJ"``)
    """

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        project_key: str,
        *,
        bug_type: str = "Bug",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._project_key = project_key
        self._bug_type = bug_type
        self._client = httpx.AsyncClient(
            base_url=f"{self._base_url}/rest/api/3",
            auth=(email, api_token),
            headers={"Accept": "application/json"},
        )

    @property
    def source_name(self) -> str:
        return "jira"

    async def fetch_updated_since(self, since: datetime) -> list[QualityEvent]:
        jql = (
            f'project = "{self._project_key}" AND issuetype = "{self._bug_type}" '
            f'AND updated >= "{since.strftime("%Y-%m-%d %H:%M")}" ORDER BY updated DESC'
        )
        return await self._search(jql)

    async def fetch_all(self) -> list[QualityEvent]:
        jql = (
            f'project = "{self._project_key}" AND issuetype = "{self._bug_type}" '
            f"ORDER BY created DESC"
        )
        return await self._search(jql)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/myself")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _search(self, jql: str) -> list[QualityEvent]:
        events: list[QualityEvent] = []
        start_at = 0
        while True:
            resp = await self._client.get(
                "/search",
                params={"jql": jql, "startAt": start_at, "maxResults": 50},
            )
            resp.raise_for_status()
            data = resp.json()
            for issue in data.get("issues", []):
                events.append(self._normalize(issue))
            if start_at + len(data.get("issues", [])) >= data.get("total", 0):
                break
            start_at += len(data.get("issues", []))
        return events

    def _normalize(self, issue: dict[str, Any]) -> QualityEvent:
        fields = issue.get("fields", {})
        description_body = ""
        if fields.get("description"):
            description_body = self._extract_text(fields["description"])

        status_name = fields.get("status", {}).get("name", "")
        priority_name = fields.get("priority", {}).get("name", "")

        return QualityEvent(
            source=EventSource.JIRA,
            external_id=issue["key"],
            title=fields.get("summary", ""),
            description=description_body,
            status=_STATUS_MAP.get(status_name, EventStatus.OPEN),
            severity=_PRIORITY_MAP.get(priority_name, Severity.MEDIUM),
            metadata={"jira_key": issue["key"], "labels": fields.get("labels", [])},
            created_at=datetime.fromisoformat(fields["created"].replace("Z", "+00:00"))
            if fields.get("created")
            else datetime.utcnow(),
        )

    @staticmethod
    def _extract_text(adf: Any) -> str:
        """Recursively extract plain text from Atlassian Document Format."""
        if isinstance(adf, str):
            return adf
        if isinstance(adf, dict):
            if adf.get("type") == "text":
                return adf.get("text", "")
            return " ".join(
                JiraIssueSourceAdapter._extract_text(child)
                for child in adf.get("content", [])
            )
        if isinstance(adf, list):
            return " ".join(JiraIssueSourceAdapter._extract_text(item) for item in adf)
        return ""
