"""
Core domain models for the Quality Knowledge Base.

Every concept in perfectCode flows through these types. They are plain data
containers (Pydantic models) with no coupling to any specific adapter, storage
engine, or LLM provider.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventSource(str, Enum):
    """Where a QualityEvent originated."""

    JIRA = "jira"
    GITHUB = "github"
    LINEAR = "linear"
    MANUAL = "manual"
    LLM_FIX = "llm_fix"
    DEVELOPER_FIX = "developer_fix"
    CUSTOM = "custom"


class EventStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    DUPLICATE = "duplicate"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TagAssignmentSource(str, Enum):
    """How a tag was assigned to an event."""

    LLM = "llm"
    MANUAL = "manual"
    RULE = "rule"


class DevEventType(str, Enum):
    """What kind of developer action triggered a validation check."""

    EDIT = "edit"
    CODE_GENERATION = "code_generation"
    TEST_FAILURE = "test_failure"
    RUNTIME_ERROR = "runtime_error"
    COMMIT = "commit"


# ---------------------------------------------------------------------------
# Quality Event — the universal issue / bug / incident record
# ---------------------------------------------------------------------------


class CodeLocation(BaseModel):
    """A pointer to a specific location in a codebase."""

    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    function_name: str | None = None
    repo: str | None = None
    branch: str | None = None


class CodePatch(BaseModel):
    """
    Actual code content associated with a bug — the buggy code (before) and
    optionally the fix (after). Extracted from linked PRs, commit diffs, issue
    descriptions, or LLM analysis.
    """

    file_path: str
    language: str | None = None
    before: str = ""
    after: str = ""
    line_start: int | None = None
    line_end: int | None = None
    source: str = ""


class QualityEvent(BaseModel):
    """
    Normalized representation of any issue, bug, incident, or fix.

    This is the fundamental unit stored in the Quality Knowledge Base. Regardless
    of whether it came from Jira, GitHub Issues, an LLM self-correction, or a
    developer's manual entry, it always looks the same.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source: EventSource
    external_id: str | None = None

    title: str
    description: str = ""
    stack_traces: list[str] = Field(default_factory=list)
    code_locations: list[CodeLocation] = Field(default_factory=list)
    code_patches: list[CodePatch] = Field(default_factory=list)

    status: EventStatus = EventStatus.OPEN
    severity: Severity = Severity.MEDIUM

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    duplicate_of: str | None = None


# ---------------------------------------------------------------------------
# Taxonomy — hierarchical tagging
# ---------------------------------------------------------------------------


class TagNode(BaseModel):
    """
    A single node in the QualityTaxonomy tree.

    Tags form a hierarchy: ``Bugs`` -> ``Bugs/Language`` -> ``Bugs/Language/Python``
    -> ``Bugs/Language/Python/TypeErrors``. Each node carries a human-readable
    description that the LLM uses when deciding whether to assign or create tags.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    parent_id: str | None = None
    path: str = ""
    children_count: int = 0

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    version: int = 1


class IssueTagAssignment(BaseModel):
    """Link between a QualityEvent and a TagNode."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_id: str
    tag_id: str

    source: TagAssignmentSource = TagAssignmentSource.LLM
    confidence: float = 1.0
    created_by: str = "system"

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Index — how events are represented inside an indexer
# ---------------------------------------------------------------------------


class IndexEntry(BaseModel):
    """
    Represents a QualityEvent as stored in a particular indexer backend.

    Each indexer may store different data (an embedding vector, an inverted-index
    document, a graph node ID, etc.), captured generically in ``index_data``.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_id: str
    indexer_name: str
    index_data: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Search — queries and results across indexers
# ---------------------------------------------------------------------------


class TagFilter(BaseModel):
    """Constrain search results to events under specific taxonomy branches."""

    tag_ids: list[str] = Field(default_factory=list)
    tag_paths: list[str] = Field(default_factory=list)
    boost: float = 1.0


class SearchQuery(BaseModel):
    """
    A search request sent to one or more IndexerAdapters.

    Can carry raw text, a code snippet, structured filters, or a combination.
    """

    text: str = ""
    code_snippet: str = ""
    language: str | None = None
    tag_filter: TagFilter | None = None
    max_results: int = 10
    min_score: float = 0.0
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single result returned by an indexer search."""

    event: QualityEvent
    score: float = 0.0
    indexer_name: str = ""
    matched_tags: list[str] = Field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Dev Context — edits and events from coding environments
# ---------------------------------------------------------------------------


class DiffHunk(BaseModel):
    """A single hunk of a unified diff."""

    file_path: str
    language: str | None = None
    before: str = ""
    after: str = ""
    line_start: int | None = None


class DevEvent(BaseModel):
    """
    An event from a developer's coding session.

    Sent by a DevContextAdapter to the ValidationService for checking against
    the Quality Knowledge Base.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: DevEventType
    diff_hunks: list[DiffHunk] = Field(default_factory=list)
    error_message: str = ""
    file_paths: list[str] = Field(default_factory=list)
    project: str = ""
    branch: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Validation — result of checking an edit against the QKB
# ---------------------------------------------------------------------------


class ValidationFinding(BaseModel):
    """One potential issue detected during edit validation."""

    related_event: QualityEvent
    similarity_score: float = 0.0
    risk_level: Severity = Severity.MEDIUM
    explanation: str = ""
    suggestion: str = ""


class ValidationResult(BaseModel):
    """Full result of validating a DevEvent against the QKB."""

    dev_event_id: str
    findings: list[ValidationFinding] = Field(default_factory=list)
    passed: bool = True
    summary: str = ""
    new_tags_suggested: list[TagNode] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# LLM — prompt configuration
# ---------------------------------------------------------------------------


class PromptTemplate(BaseModel):
    """
    Configuration for a single type of LLM call.

    Each call-type (tag suggestion, edit validation, etc.) has its own template
    and parameters. Users can override these per project and per provider.
    """

    name: str
    description: str = ""
    system_prompt: str = ""
    user_template: str = ""
    model_hint: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    extra_params: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Normalized response from any LLM provider."""

    content: str
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Meta Issue — for clustering duplicates / near-duplicates
# ---------------------------------------------------------------------------


class MetaIssue(BaseModel):
    """
    Groups related QualityEvents that are near-duplicates or describe the same
    underlying problem.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    description: str = ""
    member_event_ids: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
