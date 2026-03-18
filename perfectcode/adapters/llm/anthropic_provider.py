"""
Anthropic LLM provider adapter.

Wraps the Anthropic Python SDK (``anthropic>=0.25``) behind the common
LLMAdapter interface. Install with ``pip install perfectcode[anthropic]``.
"""

from __future__ import annotations

from typing import Any

from perfectcode.adapters.llm.base import LLMAdapter
from perfectcode.core.models import LLMResponse, PromptTemplate


class AnthropicProvider(LLMAdapter):
    """
    Calls Anthropic's Messages API (Claude).

    Requires an ``ANTHROPIC_API_KEY`` environment variable or an explicit
    ``api_key`` constructor argument.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install the anthropic extra: pip install perfectcode[anthropic]"
            ) from exc

        self._default_model = default_model
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def call(self, prompt: PromptTemplate, variables: dict[str, str]) -> LLMResponse:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        model = prompt.model_hint or self._default_model
        rendered_user = prompt.user_template.format_map(variables)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": prompt.max_tokens,
            "messages": [{"role": "user", "content": rendered_user}],
        }
        if prompt.system_prompt:
            kwargs["system"] = prompt.system_prompt

        response = await client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def health_check(self) -> bool:
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self._api_key)
            await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False
