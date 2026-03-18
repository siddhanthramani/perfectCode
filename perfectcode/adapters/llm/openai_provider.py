"""
OpenAI LLM provider adapter.

Wraps the OpenAI Python SDK (``openai>=1.30``) behind the common LLMAdapter
interface. Install with ``pip install perfectcode[openai]``.
"""

from __future__ import annotations

from typing import Any

from perfectcode.adapters.llm.base import LLMAdapter
from perfectcode.core.models import LLMResponse, PromptTemplate


class OpenAIProvider(LLMAdapter):
    """
    Calls OpenAI's chat completions API.

    Requires an ``OPENAI_API_KEY`` environment variable or an explicit
    ``api_key`` constructor argument.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o",
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install the openai extra: pip install perfectcode[openai]"
            ) from exc

        self._default_model = default_model
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "openai"

    async def call(self, prompt: PromptTemplate, variables: dict[str, str]) -> LLMResponse:
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key)
        model = prompt.model_hint or self._default_model
        rendered_user = prompt.user_template.format_map(variables)

        messages: list[dict[str, Any]] = []
        if prompt.system_prompt:
            messages.append({"role": "system", "content": prompt.system_prompt})
        messages.append({"role": "user", "content": rendered_user})

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
        )

    async def health_check(self) -> bool:
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self._api_key)
            await client.models.list()
            return True
        except Exception:
            return False
