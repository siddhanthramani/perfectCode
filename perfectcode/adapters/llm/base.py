"""
Abstract base for LLM adapters and prompt management.

The LLM layer is completely vendor-neutral. A single ``LLMAdapter`` interface
hides the specifics of OpenAI, Anthropic, local models, or any HTTP-based
provider. Prompts are configured per call-type and fully overridable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from perfectcode.core.models import LLMResponse, PromptTemplate


class LLMCallType(str, Enum):
    """
    Well-known call types that the system uses.

    Each call type has its own default prompt template and configuration.
    Users may register additional custom call types.
    """

    TAG_SUGGESTION = "tag_suggestion"
    TAXONOMY_REORG = "taxonomy_reorg"
    EDIT_VALIDATION = "edit_validation"
    SIMILARITY_EXPLAIN = "similarity_explain"
    CUSTOM = "custom"


class LLMAdapter(ABC):
    """
    Vendor-neutral interface for calling a language model.

    Implementations wrap a specific provider SDK (OpenAI, Anthropic, etc.)
    but expose the same ``call`` method. Prompt templates are resolved
    externally and passed in; the adapter only handles transport.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for the provider, e.g. ``'openai'``, ``'anthropic'``."""
        ...

    @abstractmethod
    async def call(
        self,
        prompt: PromptTemplate,
        variables: dict[str, str],
    ) -> LLMResponse:
        """
        Render *prompt* with *variables* and send it to the model.

        *variables* are substituted into ``prompt.user_template`` using
        standard Python ``str.format_map`` before sending.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the LLM provider is reachable."""
        ...


class PromptRegistry:
    """
    Stores default and user-overridden PromptTemplates by call type.

    There is one registry per project. It ships with sensible defaults and
    lets users swap templates at runtime.
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._templates[LLMCallType.TAG_SUGGESTION] = PromptTemplate(
            name="tag_suggestion",
            description="Suggest taxonomy tags for a quality event",
            system_prompt=(
                "You are a code quality taxonomy expert. Given an issue description "
                "and the current tag hierarchy, assign existing tags or propose new ones."
            ),
            user_template=(
                "## Issue\nTitle: {title}\nDescription: {description}\n\n"
                "## Current Taxonomy (relevant branch)\n{taxonomy_excerpt}\n\n"
                "Return a JSON object with:\n"
                '- "existing_tag_ids": list of tag IDs to assign\n'
                '- "new_tags": list of objects with "name", "parent_id", "description"'
            ),
            temperature=0.2,
        )
        self._templates[LLMCallType.TAXONOMY_REORG] = PromptTemplate(
            name="taxonomy_reorg",
            description="Suggest reorganization of the quality taxonomy",
            system_prompt=(
                "You are a taxonomy organization expert. Analyze the tag hierarchy "
                "statistics and suggest merges, splits, or structural changes."
            ),
            user_template=(
                "## Current Taxonomy\n{taxonomy_tree}\n\n"
                "## Usage Statistics\n{usage_stats}\n\n"
                "Suggest reorganization as a JSON list of operations: "
                "merge, split, move, rename, or create."
            ),
            temperature=0.3,
        )
        self._templates[LLMCallType.EDIT_VALIDATION] = PromptTemplate(
            name="edit_validation",
            description="Validate a code edit against known quality events",
            system_prompt=(
                "You are a code quality guardian. Compare the developer's current edit "
                "against historically similar bugs. Flag anything that looks like a "
                "reintroduction of a known issue."
            ),
            user_template=(
                "## Current Edit\n```\n{diff}\n```\n\n"
                "## Similar Past Issues\n{similar_issues}\n\n"
                "For each potential match, explain the risk and suggest a fix. "
                "Return JSON with a list of findings."
            ),
            temperature=0.1,
        )
        self._templates[LLMCallType.SIMILARITY_EXPLAIN] = PromptTemplate(
            name="similarity_explain",
            description="Explain similarity between quality events for deduplication",
            system_prompt=(
                "You are a deduplication analyst. Given two issues, determine if they "
                "describe the same underlying problem."
            ),
            user_template=(
                "## Issue A\n{issue_a}\n\n## Issue B\n{issue_b}\n\n"
                "Are these duplicates? Return JSON with 'is_duplicate' (bool), "
                "'confidence' (float 0-1), and 'explanation' (string)."
            ),
            temperature=0.1,
        )

    def get(self, call_type: str | LLMCallType) -> PromptTemplate:
        key = call_type.value if isinstance(call_type, LLMCallType) else call_type
        if key not in self._templates:
            raise KeyError(f"No prompt template registered for call type '{key}'")
        return self._templates[key].model_copy()

    def override(self, call_type: str | LLMCallType, template: PromptTemplate) -> None:
        key = call_type.value if isinstance(call_type, LLMCallType) else call_type
        self._templates[key] = template

    def register(self, call_type: str, template: PromptTemplate) -> None:
        """Register a wholly new (custom) call type."""
        self._templates[call_type] = template

    def list_call_types(self) -> list[str]:
        return list(self._templates.keys())
