"""
Abstract base for dev context adapters.

A DevContextAdapter receives signals from a developer's environment — IDE
edits, AI code generation events, test failures, CI results — and forwards
them to the ValidationService for checking against the QKB.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from perfectcode.core.models import DevEvent, ValidationResult


class DevContextAdapter(ABC):
    """
    Receive developer events and return validation results.

    Implementations may be:
      - An HTTP webhook endpoint that IDEs / LLM tools POST to.
      - A filesystem watcher for local development.
      - A CI pipeline integration that sends diffs after each push.
    """

    @abstractmethod
    async def listen(self) -> AsyncIterator[DevEvent]:
        """
        Yield DevEvents as they arrive.

        The adapter is responsible for converting source-specific payloads
        (VS Code extension messages, webhook bodies, etc.) into DevEvents.
        """
        ...
        yield  # pragma: no cover — makes this a valid async generator

    @abstractmethod
    async def send_result(self, result: ValidationResult) -> None:
        """
        Push a ValidationResult back to the originating environment.

        Depending on the adapter, this could mean returning an HTTP response,
        posting a comment on a PR, or sending a notification to the IDE.
        """
        ...

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Identifier for this adapter, e.g. ``'webhook'``, ``'vscode'``."""
        ...
