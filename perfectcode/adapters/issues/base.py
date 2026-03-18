"""
Abstract base for issue source adapters.

Every external issue tracker (Jira, GitHub Issues, Linear, etc.) is accessed
through this interface. Implementations normalize tracker-specific data into
QualityEvents so the rest of the system never depends on any single tracker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from perfectcode.core.models import QualityEvent


class IssueSourceAdapter(ABC):
    """
    Pull issues from an external tracker and normalize them.

    Subclasses implement two retrieval modes:
      - incremental (``fetch_updated_since``) for daily syncs
      - full (``fetch_all``) for bulk backfills

    Both return plain ``QualityEvent`` objects ready for ingestion.
    """

    @abstractmethod
    async def fetch_updated_since(self, since: datetime) -> list[QualityEvent]:
        """Return issues created or modified after *since*."""
        ...

    @abstractmethod
    async def fetch_all(self) -> list[QualityEvent]:
        """Return every issue available in the source (for backfill)."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the external tracker is reachable."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier for this source, e.g. ``'jira'``, ``'github'``."""
        ...
