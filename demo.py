"""
perfectCode Demo — ingest real GitHub bugs, build taxonomy, validate edits.

Usage:
    python demo.py                              # default: pallets/flask, 20 issues, mock LLM
    python demo.py --repo python/cpython        # different repo
    python demo.py --max-issues 50              # more issues
    python demo.py --provider openai            # use real LLM (needs OPENAI_API_KEY)
    python demo.py --token ghp_xxx              # GitHub token for higher rate limits

No API keys required for the default run. Works fully offline with the
smart mock LLM and unauthenticated GitHub API access.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from perfectcode.adapters.indexers.taxonomy import TaxonomyIndexerAdapter
from perfectcode.adapters.indexers.text import TextIndexerAdapter
from perfectcode.adapters.indexers.vector import VectorIndexerAdapter
from perfectcode.adapters.issues.github import GitHubIssueSourceAdapter
from perfectcode.adapters.llm.base import PromptRegistry
from perfectcode.adapters.llm.factory import create_llm_adapter
from perfectcode.core.models import DevEvent, DevEventType, DiffHunk
from perfectcode.core.taxonomy import QualityTaxonomyEngine
from perfectcode.services.ingestion import IngestionOrchestrator
from perfectcode.services.validation import ValidationService

DIVIDER = "=" * 72
THIN = "-" * 72


SIMULATED_EDITS = [
    {
        "name": "Potential null/None access",
        "description": "Accessing .email on a user object that may be None",
        "diff": DiffHunk(
            file_path="api/views.py",
            language="python",
            before=(
                "def get_user_email(user_id):\n"
                "    user = db.get_user(user_id)\n"
                "    return user.email\n"
            ),
            after=(
                "def get_user_email(user_id):\n"
                "    user = db.get_user(user_id)\n"
                "    # user might be None if account was deleted\n"
                "    return user.email\n"
            ),
        ),
    },
    {
        "name": "SQL query with string interpolation",
        "description": "Building a SQL query with f-string instead of parameterized query",
        "diff": DiffHunk(
            file_path="db/queries.py",
            language="python",
            before=(
                "def search_users(name):\n"
                '    return db.execute("SELECT * FROM users WHERE name = ?", [name])\n'
            ),
            after=(
                "def search_users(name):\n"
                '    query = f"SELECT * FROM users WHERE name = \'{name}\'"\n'
                "    return db.execute(query)\n"
            ),
        ),
    },
    {
        "name": "Off-by-one in list iteration",
        "description": "Accessing items[index] without bounds checking",
        "diff": DiffHunk(
            file_path="utils/pagination.py",
            language="python",
            before=(
                "def get_page(items, page, page_size):\n"
                "    start = page * page_size\n"
                "    end = start + page_size\n"
                "    return items[start:end]\n"
            ),
            after=(
                "def get_page(items, page, page_size):\n"
                "    start = page * page_size\n"
                "    end = start + page_size\n"
                "    # may go past len(items) and include boundary index\n"
                "    return [items[i] for i in range(start, end)]\n"
            ),
        ),
    },
]


def _print_taxonomy(engine: QualityTaxonomyEngine, indent: int = 0, node_id: str | None = None) -> None:
    """Recursively print the taxonomy tree."""
    if node_id is None:
        for root in engine.root_nodes:
            _print_taxonomy(engine, indent, root.id)
        return
    node = engine.get_node(node_id)
    if not node:
        return
    prefix = "  " * indent
    count = len(engine.get_events_for_tag(node_id, include_subtree=False))
    count_str = f" ({count} events)" if count > 0 else ""
    desc = f" — {node.description}" if node.description and indent < 2 else ""
    print(f"{prefix}{'├── ' if indent > 0 else ''}{node.name}{count_str}{desc}")
    for child in engine.get_children(node_id):
        _print_taxonomy(engine, indent + 1, child.id)


async def run_demo(repo: str, max_issues: int, provider: str, token: str | None) -> None:
    owner, name = repo.split("/")

    print(f"\n{DIVIDER}")
    print("  perfectCode — Zero-Issues Super-Fast Fully-Automated Perfect Code")
    print(f"{DIVIDER}\n")

    # ── Set up adapters ──────────────────────────────────────────────
    llm = create_llm_adapter(provider)
    prompts = PromptRegistry()
    text_indexer = TextIndexerAdapter()
    vector_indexer = VectorIndexerAdapter()
    taxonomy = QualityTaxonomyEngine()
    taxonomy_indexer = TaxonomyIndexerAdapter(taxonomy)

    all_indexers = [text_indexer, vector_indexer, taxonomy_indexer]
    all_rerankers = [taxonomy_indexer, text_indexer, vector_indexer]

    source = GitHubIssueSourceAdapter(owner, name, token=token, bug_label="bug", max_issues=max_issues)

    orchestrator = IngestionOrchestrator(
        sources=[source],
        indexers=all_indexers,
        taxonomy=taxonomy,
        llm=llm,
        prompts=prompts,
    )
    validation = ValidationService(
        indexers=all_indexers,
        llm=llm,
        prompts=prompts,
        rerankers=all_rerankers,
    )

    # ── Phase 1: Ingest ──────────────────────────────────────────────
    print(f"[1/3] Ingesting bugs from github.com/{repo} (max {max_issues} issues)...")
    print(f"      LLM provider: {llm.provider_name}")
    print()

    try:
        events = await orchestrator.run_full_backfill()
    except Exception as exc:
        print(f"  ERROR: Failed to fetch issues: {exc}")
        print("  Hint: GitHub unauthenticated rate limit is 60 req/hr.")
        print("        Pass --token YOUR_GITHUB_TOKEN for 5000 req/hr.")
        return

    if not events:
        print("  No bug-labeled issues found. Try a different repo or label.")
        return

    print(f"  Ingested {len(events)} issues.\n")
    print("  Sample issues:")
    for event in events[:5]:
        sev = event.severity.value.upper()
        print(f"    [{sev:8s}] {event.title[:65]}")
    if len(events) > 5:
        print(f"    ... and {len(events) - 5} more\n")

    # ── Phase 2: Taxonomy ────────────────────────────────────────────
    print(f"\n{THIN}")
    print("[2/3] Quality Taxonomy (auto-generated)")
    print(f"{THIN}\n")

    _print_taxonomy(taxonomy)

    # ── Phase 3: Validation ──────────────────────────────────────────
    print(f"\n{THIN}")
    print("[3/3] Simulated edit validation")
    print(f"{THIN}")

    for edit in SIMULATED_EDITS:
        print(f"\n  Edit: {edit['name']}")
        print(f"  File: {edit['diff'].file_path}")
        print(f"  {edit['description']}")
        print()

        dev_event = DevEvent(
            event_type=DevEventType.EDIT,
            diff_hunks=[edit["diff"]],
            file_paths=[edit["diff"].file_path],
        )

        result = await validation.validate(dev_event)

        if result.passed:
            print("  Result: PASSED — no similar past issues found.")
        else:
            print(f"  Result: {result.summary}")
            for finding in result.findings:
                risk = finding.risk_level.value.upper()
                print(f"    [{risk:8s}] Matches: {finding.related_event.title[:55]}")
                if finding.explanation:
                    wrapped = finding.explanation[:100]
                    print(f"              {wrapped}")
                if finding.suggestion:
                    wrapped = finding.suggestion[:100]
                    print(f"              Fix: {wrapped}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  Demo complete.")
    print()
    print(f"  Issues ingested:    {len(events)}")
    tag_count = len(taxonomy._nodes)
    print(f"  Taxonomy tags:      {tag_count}")
    print(f"  LLM provider:       {llm.provider_name}")
    print()
    print("  To use a real LLM:  python demo.py --provider openai")
    print("  To use Jira:        swap GitHubIssueSourceAdapter for JiraIssueSourceAdapter")
    print(f"{DIVIDER}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="perfectCode Demo — ingest real bugs, validate edits",
    )
    parser.add_argument(
        "--repo",
        default="pallets/flask",
        help="GitHub repo in owner/name format (default: pallets/flask)",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="Maximum number of issues to ingest (default: 20)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider: mock, openai, anthropic (default: mock or PERFECTCODE_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    args = parser.parse_args()

    asyncio.run(run_demo(args.repo, args.max_issues, args.provider, args.token))


if __name__ == "__main__":
    main()
