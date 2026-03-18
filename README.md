# perfectCode

### Zero-Issues Super-Fast Fully-Automated Perfect Code

An open-source, vendor-neutral system that continuously learns from every bug, incident, and fix your team has ever encountered — and uses that knowledge in real time to prevent the same mistakes from happening again.

Every edit. Every generation. Every commit. Checked against everything you've already learned.

---

## How It Works

perfectCode is built around three core ideas:

**1. Quality Knowledge Base (QKB)**
A unified store for every issue your team has seen — pulled from Jira, GitHub Issues, Linear, or any tracker, and also from bugs caught and fixed during AI-assisted coding. Not a "bug database" — a living knowledge graph that grows with every incident and every fix.

**2. Quality Taxonomy**
A hierarchical tag system that organizes issues from broad categories (e.g., `Bugs`) down to fine-grained patterns (e.g., `Bugs/Language/Python/TypeErrors/NoneComparison`). Tags are automatically created, assigned, and reorganized by LLMs. The taxonomy is fed back into prompts so the system gets smarter at categorizing over time.

**3. Per-Edit Validation Loop**
While you code — whether manually, with Cursor, Claude Code, Codex, or any AI tool — every significant edit is compared against relevant past issues. An LLM checks your current diff against historically similar bugs and flags anything that looks like a reintroduction. The result: code that learns from your team's entire history.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Issue Source Adapters                       │
│              (Jira, GitHub Issues, Linear, Custom)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ingestion Orchestrator                        │
│          (daily sync, bulk backfill, normalization)              │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌────────────────────────────────┐
│   Indexer Adapters   │          │    Quality Taxonomy Engine     │
│  (vector, text,      │◄────────►│  (hierarchical tags, LLM-     │
│   graph, agentic)    │          │   assisted tagging & reorg)   │
└─────────┬───────────┘          └──────────────┬─────────────────┘
          │                                     │
          │         ┌───────────────────┐       │
          └────────►│  LLM Adapter      │◄──────┘
                    │  (OpenAI,         │
                    │   Anthropic,      │
                    │   local, custom)  │
                    └────────┬──────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Validation Service                            │
│     (per-edit check against QKB + taxonomy, prompt-driven)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Dev Context Adapters                           │
│     (IDE plugins, CLI hooks, CI pipelines, LLM tool hooks)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Adapters & Extensibility

Every integration point is an abstract adapter. Swap implementations without touching core logic.

| Layer | Adapter | Purpose |
|---|---|---|
| **Issue Sources** | `IssueSourceAdapter` | Pull issues from any tracker (Jira, GitHub, Linear, etc.) |
| **Indexers** | `IndexerAdapter` | Store and search issues (text/BM25, vector embeddings, graph, agentic) |
| **LLM** | `LLMAdapter` | Call any LLM provider with configurable prompts per task |
| **Dev Context** | `DevContextAdapter` | Receive edits from IDEs, AI coding tools, CI pipelines |

### Prompt Configurability

Every LLM call type has its own prompt template and configuration:

- **Tag Suggestion** — assigns taxonomy tags to new issues
- **Taxonomy Reorganization** — periodically restructures the tag hierarchy
- **Edit Validation** — checks current code changes against past issues
- **Similarity Explanation** — summarizes duplicates and near-duplicates

Templates, model selection, temperature, and all parameters are overridable per project and per provider.

---

## Why This Leads to Perfect Code

```
Bug happens → gets indexed → tagged → searchable
     ↓
Developer writes code → edit validated against past bugs
     ↓
AI catches potential reintroduction → developer fixes early
     ↓
Fix itself becomes a new QKB entry → system gets smarter
     ↓
Repeat. Every bug makes future code better.
```

The flywheel effect: the more bugs you find, the fewer bugs you ship. Over time, the system accumulates so much knowledge that entire classes of bugs become impossible to reintroduce.

---

## Project Structure

```
perfectcode/
├── core/               # Domain models and taxonomy engine
│   ├── models.py       # QualityEvent, TagNode, SearchQuery, etc.
│   └── taxonomy.py     # QualityTaxonomyEngine
├── adapters/
│   ├── issues/         # IssueSourceAdapter implementations
│   ├── indexers/       # IndexerAdapter implementations
│   ├── llm/            # LLMAdapter + provider implementations
│   └── dev/            # DevContextAdapter implementations
├── services/
│   ├── ingestion.py    # IngestionOrchestrator
│   ├── validation.py   # ValidationService
│   └── dedup.py        # Deduplication & similarity clustering
└── api/
    └── routes.py       # HTTP API (FastAPI)
```

---

## Getting Started

```bash
pip install -e ".[dev]"
```

Configure your adapters in a YAML or environment-based config, point at your issue tracker and preferred LLM provider, and run the ingestion pipeline. See individual adapter docs for setup details.

---

## Roadmap

- **v0** — Architecture design and reference interfaces (current)
- **v1** — Minimal Python service with SQLite FTS indexer, one issue source adapter, and one LLM provider
- **v2** — Vector indexer, taxonomy reorganization, IDE plugin for Cursor/VS Code
- **v3** — Graph indexer, agentic search, CI/CD integration, multi-project support

---

## License

Open source. Vendor neutral. Adapter-driven. Build what you need.
