"""
Abstract base for code enrichers.

A CodeEnricher takes a bare QualityEvent (title + description only) and
fills in structured code data: code locations, stack traces, and actual
code patches (before/after from fix diffs).

Multiple enrichers run in a configurable pipeline. Each enricher adds what
it can find; the pipeline controls order, fallback, and stop conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from perfectcode.core.models import CodeLocation, CodePatch


class EnrichmentContext(BaseModel):
    """
    Source-specific context passed to enrichers so they can follow links.

    Carries HTTP clients, repo identifiers, and anything else an enricher
    needs to reach back into the issue source's API.
    """

    source_type: str = ""
    owner: str = ""
    repo: str = ""
    base_url: str = ""
    auth_headers: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class EnrichmentResult(BaseModel):
    """What a single enricher found."""

    code_locations: list[CodeLocation] = Field(default_factory=list)
    code_patches: list[CodePatch] = Field(default_factory=list)
    stack_traces: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    enricher_name: str = ""


class EnrichmentStrategy(str, Enum):
    """Controls how multiple enrichers interact in a pipeline."""

    SUPPLEMENT = "supplement"
    FIRST_SUFFICIENT = "first_sufficient"
    ALL = "all"


class CodeEnricher(ABC):
    """
    Extracts code locations, patches, and stack traces from a QualityEvent.

    Implementations may:
      - Follow linked PRs / commits (source-specific)
      - Parse the issue description with regex (source-agnostic)
      - Call an LLM for structured extraction (source-agnostic)
    """

    @property
    @abstractmethod
    def enricher_name(self) -> str:
        """Identifier for this enricher, e.g. ``'github_pr'``, ``'description_parser'``."""
        ...

    @abstractmethod
    async def enrich(
        self,
        event: "QualityEvent",
        context: EnrichmentContext,
    ) -> EnrichmentResult:
        """
        Attempt to extract code data from *event* using *context*.

        Returns an ``EnrichmentResult`` with whatever was found. A confidence
        of 0.0 means nothing useful was extracted.
        """
        ...

    def can_handle(self, context: EnrichmentContext) -> bool:
        """Return True if this enricher can work with the given context.

        Source-specific enrichers should check ``context.source_type``.
        Source-agnostic enrichers can always return True.
        """
        return True


# avoid circular import at type-check time
from perfectcode.core.models import QualityEvent as QualityEvent  # noqa: E402, F811
