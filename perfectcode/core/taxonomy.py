"""
Quality Taxonomy Engine — hierarchical tagging with LLM-driven lifecycle.

The taxonomy is a tree of TagNodes that classifies QualityEvents from broad
categories (``Bugs``) down to very specific patterns
(``Bugs/Language/Python/TypeErrors/NoneComparison``).

This engine manages the tree structure, tag assignments, LLM-assisted tagging,
and periodic reorganization. It is exposed to the search pipeline via
``TaxonomyIndexerAdapter`` (in ``adapters/indexers/taxonomy.py``), which wraps
this engine behind both the ``IndexerAdapter`` and ``RerankerAdapter``
interfaces.

Key design decisions:
  - Tags are versioned. When the taxonomy is reorganized (merge, split, move),
    historical assignments remain traceable via their version numbers.
  - Tag creation, assignment, and reorganization are all LLM-assisted but the
    engine validates and applies changes — the LLM never writes directly.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from perfectcode.core.models import (
    IssueTagAssignment,
    QualityEvent,
    TagAssignmentSource,
    TagNode,
)

logger = logging.getLogger(__name__)


class QualityTaxonomyEngine:
    """
    Manages the full lifecycle of the hierarchical tag tree.

    Responsibilities:
      - Store and retrieve TagNodes in a tree structure.
      - Auto-tag QualityEvents by prompting an LLM with the relevant taxonomy
        branch and the event content.
      - Periodically reorganize the tree (merge, split, move, rename) via LLM.
      - Track tag assignments with source attribution and version history.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TagNode] = {}
        self._assignments: list[IssueTagAssignment] = []
        self._version: int = 1
        self._init_root()

    def _init_root(self) -> None:
        root = TagNode(
            name="Bugs",
            description="Root category for all quality events and known issues.",
            path="Bugs",
        )
        self._nodes[root.id] = root

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    @property
    def root_nodes(self) -> list[TagNode]:
        return [n for n in self._nodes.values() if n.parent_id is None]

    def get_node(self, node_id: str) -> TagNode | None:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> list[TagNode]:
        return [n for n in self._nodes.values() if n.parent_id == node_id]

    def get_subtree(self, node_id: str) -> list[TagNode]:
        """Return *node_id* and all its descendants, breadth-first."""
        result: list[TagNode] = []
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            node = self._nodes.get(current)
            if node:
                result.append(node)
                queue.extend(c.id for c in self.get_children(current))
        return result

    def find_by_path(self, path: str) -> TagNode | None:
        for node in self._nodes.values():
            if node.path == path:
                return node
        return None

    def add_node(self, name: str, description: str, parent_id: str | None = None) -> TagNode:
        """Create and register a new tag node under *parent_id*."""
        parent = self._nodes.get(parent_id) if parent_id else None
        if parent_id and not parent:
            raise ValueError(f"Parent node '{parent_id}' not found")

        path = f"{parent.path}/{name}" if parent else name
        node = TagNode(
            name=name,
            description=description,
            parent_id=parent_id,
            path=path,
            version=self._version,
        )
        self._nodes[node.id] = node

        if parent:
            parent.children_count += 1

        logger.info("Added tag node: %s (path=%s)", name, path)
        return node

    def move_node(self, node_id: str, new_parent_id: str | None) -> None:
        """Move *node_id* under *new_parent_id*, updating paths for the subtree."""
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' not found")

        old_parent = self._nodes.get(node.parent_id) if node.parent_id else None
        if old_parent:
            old_parent.children_count = max(0, old_parent.children_count - 1)

        new_parent = self._nodes.get(new_parent_id) if new_parent_id else None
        if new_parent_id and not new_parent:
            raise ValueError(f"New parent '{new_parent_id}' not found")

        old_prefix = node.path
        new_prefix = f"{new_parent.path}/{node.name}" if new_parent else node.name
        node.parent_id = new_parent_id
        node.path = new_prefix
        node.version = self._version

        if new_parent:
            new_parent.children_count += 1

        for descendant in self.get_subtree(node_id)[1:]:
            descendant.path = descendant.path.replace(old_prefix, new_prefix, 1)
            descendant.version = self._version

    def merge_nodes(self, keep_id: str, remove_id: str) -> None:
        """Merge *remove_id* into *keep_id*, reassigning all events and children."""
        keep = self._nodes.get(keep_id)
        remove = self._nodes.get(remove_id)
        if not keep or not remove:
            raise ValueError("Both nodes must exist for merge")

        for child in self.get_children(remove_id):
            self.move_node(child.id, keep_id)

        for assignment in self._assignments:
            if assignment.tag_id == remove_id:
                assignment.tag_id = keep_id

        parent = self._nodes.get(remove.parent_id) if remove.parent_id else None
        if parent:
            parent.children_count = max(0, parent.children_count - 1)

        del self._nodes[remove_id]
        logger.info("Merged tag '%s' into '%s'", remove.name, keep.name)

    def rename_node(self, node_id: str, new_name: str) -> None:
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' not found")
        old_name = node.name
        node.name = new_name
        old_path = node.path
        parts = old_path.rsplit("/", 1)
        new_path = f"{parts[0]}/{new_name}" if len(parts) > 1 else new_name
        node.path = new_path
        node.version = self._version
        for descendant in self.get_subtree(node_id)[1:]:
            descendant.path = descendant.path.replace(old_path, new_path, 1)
            descendant.version = self._version
        logger.info("Renamed tag '%s' -> '%s'", old_name, new_name)

    # ------------------------------------------------------------------
    # Tag assignment
    # ------------------------------------------------------------------

    def assign_tag(
        self,
        event_id: str,
        tag_id: str,
        source: TagAssignmentSource = TagAssignmentSource.LLM,
        confidence: float = 1.0,
    ) -> IssueTagAssignment:
        if tag_id not in self._nodes:
            raise ValueError(f"Tag '{tag_id}' not found")

        for a in self._assignments:
            if a.event_id == event_id and a.tag_id == tag_id:
                return a

        assignment = IssueTagAssignment(
            event_id=event_id,
            tag_id=tag_id,
            source=source,
            confidence=confidence,
        )
        self._assignments.append(assignment)
        return assignment

    def get_event_tags(self, event_id: str) -> list[TagNode]:
        tag_ids = {a.tag_id for a in self._assignments if a.event_id == event_id}
        return [self._nodes[tid] for tid in tag_ids if tid in self._nodes]

    def get_events_for_tag(self, tag_id: str, include_subtree: bool = False) -> list[str]:
        """Return event IDs assigned to *tag_id* (and optionally its subtree)."""
        if include_subtree:
            tag_ids = {n.id for n in self.get_subtree(tag_id)}
        else:
            tag_ids = {tag_id}
        return [a.event_id for a in self._assignments if a.tag_id in tag_ids]

    # ------------------------------------------------------------------
    # LLM-assisted auto-tagging
    # ------------------------------------------------------------------

    async def auto_tag(
        self,
        event: QualityEvent,
        llm: Any,
        prompts: Any,
    ) -> list[IssueTagAssignment]:
        """
        Use an LLM to suggest tags for *event* given the current taxonomy.

        The LLM receives a serialized excerpt of the relevant taxonomy branch
        and the event's content, and returns tag assignments and optionally
        new tag proposals.
        """
        from perfectcode.adapters.llm.base import LLMCallType

        taxonomy_excerpt = self._serialize_for_prompt()
        prompt = prompts.get(LLMCallType.TAG_SUGGESTION)
        response = await llm.call(
            prompt,
            {
                "title": event.title,
                "description": event.description,
                "taxonomy_excerpt": taxonomy_excerpt,
            },
        )

        return self._apply_tag_response(event.id, response.content)

    def _serialize_for_prompt(self, max_depth: int = 5) -> str:
        """Render the taxonomy tree as indented text for LLM consumption."""
        lines: list[str] = []

        def _walk(node_id: str, depth: int) -> None:
            if depth > max_depth:
                return
            node = self._nodes[node_id]
            indent = "  " * depth
            desc = f" — {node.description}" if node.description else ""
            lines.append(f"{indent}- [{node.id}] {node.name}{desc}")
            for child in self.get_children(node_id):
                _walk(child.id, depth + 1)

        for root in self.root_nodes:
            _walk(root.id, 0)

        return "\n".join(lines) if lines else "(empty taxonomy)"

    def _apply_tag_response(self, event_id: str, llm_output: str) -> list[IssueTagAssignment]:
        """Parse the LLM's tag suggestions and apply them."""
        assignments: list[IssueTagAssignment] = []
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for tag suggestion; skipping")
            return assignments

        for tag_id in data.get("existing_tag_ids", []):
            if tag_id in self._nodes:
                assignments.append(self.assign_tag(event_id, tag_id))

        for new_tag in data.get("new_tags", []):
            name = new_tag.get("name", "")
            parent_id = new_tag.get("parent_id")
            description = new_tag.get("description", "")
            if not name:
                continue
            try:
                node = self.add_node(name, description, parent_id)
                assignments.append(self.assign_tag(event_id, node.id))
            except ValueError:
                logger.warning("Failed to create new tag '%s' under '%s'", name, parent_id)

        return assignments

    # ------------------------------------------------------------------
    # LLM-assisted reorganization
    # ------------------------------------------------------------------

    async def reorganize(self, llm: Any, prompts: Any) -> list[str]:
        """
        Prompt an LLM to suggest structural changes to the taxonomy.

        Returns a list of human-readable descriptions of changes applied.
        """
        from perfectcode.adapters.llm.base import LLMCallType

        prompt = prompts.get(LLMCallType.TAXONOMY_REORG)
        response = await llm.call(
            prompt,
            {
                "taxonomy_tree": self._serialize_for_prompt(max_depth=10),
                "usage_stats": self._compute_usage_stats(),
            },
        )

        return self._apply_reorg_response(response.content)

    def _compute_usage_stats(self) -> str:
        """Build a summary of tag usage for the reorganization prompt."""
        counts: dict[str, int] = defaultdict(int)
        for a in self._assignments:
            counts[a.tag_id] += 1

        lines: list[str] = []
        for node_id, node in self._nodes.items():
            count = counts.get(node_id, 0)
            lines.append(f"- {node.path}: {count} events, {node.children_count} children")
        return "\n".join(lines) if lines else "(no usage data)"

    def _apply_reorg_response(self, llm_output: str) -> list[str]:
        """Parse and apply reorganization operations from the LLM."""
        applied: list[str] = []
        try:
            operations = json.loads(llm_output)
            if not isinstance(operations, list):
                operations = operations.get("operations", [])
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for reorg; skipping")
            return applied

        self._version += 1

        for op in operations:
            op_type = op.get("type", "")
            try:
                match op_type:
                    case "merge":
                        self.merge_nodes(op["keep_id"], op["remove_id"])
                        applied.append(f"Merged '{op['remove_id']}' into '{op['keep_id']}'")
                    case "move":
                        self.move_node(op["node_id"], op.get("new_parent_id"))
                        applied.append(f"Moved '{op['node_id']}' to '{op.get('new_parent_id')}'")
                    case "rename":
                        self.rename_node(op["node_id"], op["new_name"])
                        applied.append(f"Renamed '{op['node_id']}' to '{op['new_name']}'")
                    case "create":
                        self.add_node(op["name"], op.get("description", ""), op.get("parent_id"))
                        applied.append(f"Created tag '{op['name']}'")
                    case "split":
                        applied.append(f"Split requested for '{op.get('node_id')}' (manual review)")
                    case _:
                        logger.warning("Unknown reorg operation type: %s", op_type)
            except (ValueError, KeyError) as exc:
                logger.warning("Failed to apply reorg operation %s: %s", op, exc)

        logger.info("Taxonomy reorganized (v%d): %d operations applied", self._version, len(applied))
        return applied

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "nodes": {nid: n.model_dump() for nid, n in self._nodes.items()},
            "assignments": [a.model_dump() for a in self._assignments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityTaxonomyEngine:
        engine = cls.__new__(cls)
        engine._version = data.get("version", 1)
        engine._nodes = {
            nid: TagNode(**ndata) for nid, ndata in data.get("nodes", {}).items()
        }
        engine._assignments = [
            IssueTagAssignment(**a) for a in data.get("assignments", [])
        ]
        return engine
