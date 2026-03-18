"""
LLM provider factory.

Reads ``PERFECTCODE_LLM_PROVIDER`` (or an explicit argument) and returns
the matching ``LLMAdapter``. Defaults to the smart mock so the system works
fully offline with zero API keys.

Supported values:
  - ``"mock"``      — keyword-driven mock (default, no API key)
  - ``"openai"``    — OpenAI (requires ``OPENAI_API_KEY``)
  - ``"anthropic"`` — Anthropic (requires ``ANTHROPIC_API_KEY``)
"""

from __future__ import annotations

import os

from perfectcode.adapters.llm.base import LLMAdapter
from perfectcode.adapters.llm.mock import MockLLMAdapter


def create_llm_adapter(provider: str | None = None) -> LLMAdapter:
    """
    Build and return the LLM adapter for *provider*.

    Falls back to the ``PERFECTCODE_LLM_PROVIDER`` environment variable,
    then to ``"mock"``.
    """
    name = (provider or os.environ.get("PERFECTCODE_LLM_PROVIDER", "mock")).lower().strip()

    if name == "mock":
        return MockLLMAdapter()

    if name == "openai":
        from perfectcode.adapters.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()

    if name == "anthropic":
        from perfectcode.adapters.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()

    raise ValueError(
        f"Unknown LLM provider '{name}'. "
        f"Supported: mock, openai, anthropic. "
        f"Set PERFECTCODE_LLM_PROVIDER or pass --provider."
    )
