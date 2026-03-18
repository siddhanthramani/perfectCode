"""
HTTP API layer (FastAPI).

Exposes the Quality Knowledge Base, taxonomy, and validation service over HTTP.
This is the integration surface that IDE plugins, CI tools, and LLM coding
environments call.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from perfectcode.adapters.dev.webhook import WebhookDevContextAdapter
from perfectcode.adapters.indexers.taxonomy import TaxonomyIndexerAdapter
from perfectcode.adapters.indexers.text import TextIndexerAdapter
from perfectcode.adapters.indexers.vector import VectorIndexerAdapter
from perfectcode.adapters.llm.base import PromptRegistry
from perfectcode.adapters.llm.mock import MockLLMAdapter
from perfectcode.core.models import DevEvent, SearchQuery, ValidationResult
from perfectcode.core.taxonomy import QualityTaxonomyEngine
from perfectcode.services.ingestion import IngestionOrchestrator
from perfectcode.services.validation import ValidationService

app = FastAPI(
    title="perfectCode",
    description="Zero-Issues Super-Fast Fully-Automated Perfect Code",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Bootstrap default components (swap via config in production)
# ---------------------------------------------------------------------------

_text_indexer = TextIndexerAdapter()
_vector_indexer = VectorIndexerAdapter()
_llm = MockLLMAdapter()
_prompts = PromptRegistry()
_taxonomy = QualityTaxonomyEngine()
_taxonomy_indexer = TaxonomyIndexerAdapter(_taxonomy)
_dev_adapter = WebhookDevContextAdapter()

_all_indexers = [_text_indexer, _vector_indexer, _taxonomy_indexer]
_all_rerankers = [_taxonomy_indexer, _text_indexer, _vector_indexer]

_validation = ValidationService(
    indexers=_all_indexers,
    llm=_llm,
    prompts=_prompts,
    rerankers=_all_rerankers,
)
_orchestrator = IngestionOrchestrator(
    sources=[],
    indexers=_all_indexers,
    taxonomy=_taxonomy,
    llm=_llm,
    prompts=_prompts,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    since: datetime | None = None
    full_backfill: bool = False


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, Any]:
    if req.full_backfill:
        events = await _orchestrator.run_full_backfill()
    else:
        events = await _orchestrator.run_incremental(since=req.since)
    return {"ingested": len(events), "event_ids": [e.id for e in events]}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@app.post("/search")
async def search(query: SearchQuery) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for indexer in _all_indexers:
        for r in await indexer.search(query):
            if r.event.id not in seen:
                seen.add(r.event.id)
                all_results.append(
                    {
                        "event_id": r.event.id,
                        "title": r.event.title,
                        "score": r.score,
                        "indexer": r.indexer_name,
                        "matched_tags": r.matched_tags,
                    }
                )
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[: query.max_results]


# ---------------------------------------------------------------------------
# Validation (per-edit)
# ---------------------------------------------------------------------------


@app.post("/validate", response_model=ValidationResult)
async def validate(event: DevEvent) -> ValidationResult:
    return await _validation.validate(event)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


@app.get("/taxonomy")
async def get_taxonomy() -> dict[str, Any]:
    return _taxonomy.to_dict()


class AddTagRequest(BaseModel):
    name: str
    description: str = ""
    parent_id: str | None = None


@app.post("/taxonomy/tags")
async def add_tag(req: AddTagRequest) -> dict[str, Any]:
    try:
        node = _taxonomy.add_node(req.name, req.description, req.parent_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return node.model_dump()


@app.post("/taxonomy/reorganize")
async def reorganize_taxonomy() -> dict[str, Any]:
    changes = await _taxonomy.reorganize(_llm, _prompts)
    return {"changes_applied": changes}
