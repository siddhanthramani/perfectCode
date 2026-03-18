"""
Mock issue source adapter for testing and development.

Returns a handful of synthetic QualityEvents so the ingestion pipeline
can be exercised without any external dependency.
"""

from __future__ import annotations

from datetime import datetime

from perfectcode.adapters.issues.base import IssueSourceAdapter
from perfectcode.core.models import (
    CodeLocation,
    EventSource,
    EventStatus,
    QualityEvent,
    Severity,
)

_SAMPLE_EVENTS = [
    QualityEvent(
        source=EventSource.MANUAL,
        external_id="MOCK-1",
        title="NoneType attribute access in user serializer",
        description=(
            "Calling .email on a User object that can be None when the account "
            "is deactivated. Causes AttributeError in the /api/users endpoint."
        ),
        stack_traces=["AttributeError: 'NoneType' object has no attribute 'email'"],
        code_locations=[CodeLocation(file_path="api/serializers.py", line_start=42)],
        status=EventStatus.RESOLVED,
        severity=Severity.HIGH,
    ),
    QualityEvent(
        source=EventSource.MANUAL,
        external_id="MOCK-2",
        title="Off-by-one in pagination cursor",
        description=(
            "The cursor-based pagination returns the last item of the previous "
            "page as the first item of the next page, leading to duplicates."
        ),
        code_locations=[CodeLocation(file_path="api/pagination.py", line_start=88)],
        status=EventStatus.RESOLVED,
        severity=Severity.MEDIUM,
    ),
    QualityEvent(
        source=EventSource.MANUAL,
        external_id="MOCK-3",
        title="SQL injection via unsanitized search parameter",
        description=(
            "The search query parameter is interpolated directly into a raw SQL "
            "query string without parameterization."
        ),
        code_locations=[CodeLocation(file_path="db/queries.py", line_start=15)],
        status=EventStatus.RESOLVED,
        severity=Severity.CRITICAL,
    ),
]


class MockIssueSourceAdapter(IssueSourceAdapter):
    """Returns canned QualityEvents for local development and tests."""

    async def fetch_updated_since(self, since: datetime) -> list[QualityEvent]:
        return list(_SAMPLE_EVENTS)

    async def fetch_all(self) -> list[QualityEvent]:
        return list(_SAMPLE_EVENTS)

    async def health_check(self) -> bool:
        return True

    @property
    def source_name(self) -> str:
        return "mock"
