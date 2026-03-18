"""
Webhook-based dev context adapter.

Provides a simple push model: external systems (IDE plugins, CI, LLM tools)
POST DevEvents to an HTTP endpoint, and this adapter yields them for
validation. Results are stored in-memory for retrieval.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from perfectcode.adapters.dev.base import DevContextAdapter
from perfectcode.core.models import DevEvent, ValidationResult


class WebhookDevContextAdapter(DevContextAdapter):
    """
    Receives DevEvents pushed into an asyncio queue and yields them.

    External HTTP handlers (e.g. FastAPI routes) call ``push_event`` to
    enqueue; the ValidationService consumes via ``listen``.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[DevEvent] = asyncio.Queue()
        self._results: dict[str, ValidationResult] = {}

    async def push_event(self, event: DevEvent) -> None:
        await self._queue.put(event)

    async def listen(self) -> AsyncIterator[DevEvent]:
        while True:
            event = await self._queue.get()
            yield event

    async def send_result(self, result: ValidationResult) -> None:
        self._results[result.dev_event_id] = result

    def get_result(self, dev_event_id: str) -> ValidationResult | None:
        return self._results.get(dev_event_id)

    @property
    def adapter_name(self) -> str:
        return "webhook"
