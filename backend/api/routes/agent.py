"""Agent-facing API route: the final, high-level entry point into OmniFlow.

Unlike /ingest (a lower-level endpoint that only normalizes inputs), this
route accepts a user query plus optional files, ingests the files through
the existing ingestion layer, builds a fresh AgentState, executes the full
LangGraph workflow (intent -> clarity check -> plan/execute/observe/replan
-> synthesize -> validate), and returns the structured OmniFlowResponse.

Every request gets its own fresh RAGService/ToolRegistry -- no index or
tool state persists across requests (no persistent memory yet, per Phase 4
scope). RAG indexing is LAZY (Phase 5 hardening): this route hands the
request's normalized_documents to RAGSearchTool but never calls
RAGService.index_documents() itself -- the tool indexes them only on its
own first invocation, i.e. only if the planner actually selects rag_search.
A direct-context request never loads the embedding backend just because
files were uploaded. This route owns no orchestration logic itself; it
only ingests, constructs state, calls run_graph(), and reshapes the result.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from functools import lru_cache

from backend.agents.cross_source import analyze_cross_source
from backend.agents.intent import IntentType
from backend.agents.provenance import build_evidence_references
from backend.api.routes.ingest import get_ingestion_service
from backend.exceptions import ConfigurationError, InvalidInputError, OmniFlowException
from backend.graph import run_graph
from backend.models.document import NormalizedDocument
from backend.models.request import UploadedInput
from backend.models.response import OmniFlowResponse
from backend.models.state import AgentState
from backend.models.trace import ExecutionTrace
from backend.providers.base import BaseLLMProvider
from backend.providers.gemini_provider import GeminiProvider
from backend.rag.embeddings import EmbeddingService
from backend.rag.service import RAGService
from backend.services.ingestion import IngestionService
from backend.tools.rag_search import RAGSearchTool
from backend.tools.registry import ToolRegistry
from backend.tools.youtube import YouTubeTranscriptTool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agent"])


def get_llm_provider() -> BaseLLMProvider | ConfigurationError:
    """Dependency provider for the BaseLLMProvider used by the workflow.

    A fresh GeminiProvider per request (lightweight, no network call at
    construction -- see GeminiProvider's own docstring). Tests override
    this dependency with a mock provider; no real Gemini call is made in
    the test suite.

    Returns the caught ConfigurationError instead of raising it when
    GeminiProvider cannot be constructed (e.g. GEMINI_API_KEY unset).
    FastAPI resolves every declared Depends() before the route body ever
    runs, so a raised exception here would preempt FastAPI's own request
    validation -- confirmed empirically: even a request missing the
    required `query` field entirely was returning a 500 CONFIGURATION_ERROR
    instead of a 422, since this dependency's failure short-circuited
    dependency resolution before `query` was ever validated. query_agent()
    re-raises this only after confirming the request itself is valid, so a
    server configuration problem never masks the client's own mistake.
    """
    try:
        return GeminiProvider()
    except ConfigurationError as exc:
        return exc


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Process-wide cached embedding model (same pattern as config.get_settings).

    The sentence-transformers model is a stateless text->vector encoder --
    safe to share across requests -- but is expensive to load from disk
    (~80MB). Without this cache, get_rag_service() constructing a fresh
    RAGService() per request would reload the model from scratch on every
    single request that touches RAG, even ones that never end up calling
    rag_search. Cached at most once per process; never holds any
    request-specific document or index state itself.
    """
    return EmbeddingService()


def get_rag_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RAGService:
    """Dependency provider for a fresh, request-scoped RAGService.

    The FAISS index/chunk registry are NOT shared or cached: each request
    indexes only its own just-ingested documents, so no document from one
    request's corpus is ever retrievable from another's (no persistent
    memory yet). Only the underlying embedding model (see
    get_embedding_service) is reused across requests.
    """
    return RAGService(embedding_service=embedding_service)


def _build_tool_registry(
    rag_service: RAGService, normalized_documents: list[NormalizedDocument]
) -> ToolRegistry:
    """Builds the ToolRegistry the graph selects tools from.

    Bundles every tool the planner is allowed to select: retrieval (bound
    to this request's own RAGService AND this request's own just-ingested
    documents) and YouTube transcript retrieval (stateless).

    Not a FastAPI Depends() provider: it needs ``normalized_documents``,
    which only exists after ingestion runs inside the route body --
    FastAPI resolves every Depends() before that body ever executes, so
    this must be a plain function called from within it (see query_agent()).
    Constructing RAGSearchTool with these documents does NOT index them --
    indexing is deferred to the tool's own first invocation (see
    RAGSearchTool's docstring) so a direct-context request never loads the
    embedding backend just because files were uploaded.
    """
    registry = ToolRegistry()
    registry.register(RAGSearchTool(rag_service, normalized_documents))
    registry.register(YouTubeTranscriptTool())
    return registry


@router.post("/query", response_model=OmniFlowResponse)
def query_agent(
    query: Annotated[str, Form(description="The user's question or instruction.")],
    session_id: Annotated[
        str | None, Form(description="Optional session or conversation identifier.")
    ] = None,
    files: Annotated[
        list[UploadFile] | None,
        File(description="Optional list of file uploads (PDF, image, audio)."),
    ] = None,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    llm_provider: BaseLLMProvider | ConfigurationError = Depends(get_llm_provider),
    rag_service: RAGService = Depends(get_rag_service),
) -> OmniFlowResponse:
    """Runs the full agent workflow for a user query plus optional files.

    Ingests any uploaded files via the existing IngestionService, constructs
    an AgentState, and executes the LangGraph workflow end to end -- intent
    understanding, ambiguity check (and, if ambiguous, an early stop with a
    clarifying question), bounded plan/execute/observe/replan, synthesis,
    and structural output validation.

    RAG indexing is LAZY: this route never calls
    rag_service.index_documents() itself. It hands this request's
    normalized_documents to a fresh RAGSearchTool (see _build_tool_registry),
    which indexes them only on its own first invocation -- i.e. only if the
    planner actually selects rag_search. A direct-context request (e.g.
    "summarize this PDF" answered straight from unified_context) never
    triggers indexing, so it never loads the embedding backend
    (sentence-transformers/Torch) at all.
    """
    if not query or not query.strip():
        raise InvalidInputError("query must not be empty.")

    uploaded_inputs: list[UploadedInput] = []
    file_payloads: list[tuple[bytes, str, str | None]] = []

    if files:
        for file in files:
            filename = file.filename or "upload"
            content = file.file.read()
            declared_mime = file.content_type
            file_payloads.append((content, filename, declared_mime))
            uploaded_inputs.append(
                UploadedInput(
                    filename=filename,
                    mime_type=declared_mime or "application/octet-stream",
                    size_bytes=len(content),
                )
            )

    normalized_docs = ingestion_service.ingest_inputs(
        files=file_payloads if file_payloads else None
    )

    tool_registry = _build_tool_registry(rag_service, normalized_docs)

    # Checked only after every client-attributable input error (blank
    # query, unsupported/oversized/corrupt files, OCR/transcription
    # failures) has already had its chance to raise its own specific,
    # actionable error above. A server configuration problem is real, but
    # it must never mask a mistake that is actually the client's to fix --
    # see get_llm_provider()'s docstring for why this can't just be a
    # raised exception from a Depends().
    if isinstance(llm_provider, ConfigurationError):
        raise llm_provider

    initial_state = AgentState(
        original_request=query,
        session_id=session_id,
        uploaded_inputs=uploaded_inputs,
        normalized_documents=normalized_docs,
    )

    final_state = run_graph(
        initial_state, tool_registry=tool_registry, llm_provider=llm_provider
    )

    all_warnings = list(final_state.warnings)
    for doc in normalized_docs:
        all_warnings.extend(doc.warnings)

    total_duration_ms = sum(
        step.duration_ms for step in final_state.execution_trace if step.duration_ms is not None
    )

    evidence = build_evidence_references(final_state)

    # Cross-source analysis (see backend/agents/cross_source.py): a single
    # extra structured LLM call, run only for comparison-style requests
    # over 2+ ingested sources -- never inside the graph itself, so
    # planning/tool selection are completely unaffected. A failure here is
    # surfaced as a warning, not a hard failure of the whole request: the
    # main answer already completed successfully by this point.
    cross_source_analysis = None
    if (
        len(normalized_docs) >= 2
        and final_state.intent_result is not None
        and final_state.intent_result.intent == IntentType.COMPARISON
    ):
        try:
            cross_source_analysis = analyze_cross_source(final_state, llm_provider)
        except OmniFlowException as exc:
            all_warnings.append(
                f"Cross-source analysis unavailable: {exc.error_code}: {exc.message}"
            )

    return OmniFlowResponse(
        session_id=final_state.session_id,
        status=final_state.status.value,
        answer=final_state.final_answer,
        clarification_needed=final_state.clarification_needed,
        clarification_prompt=final_state.clarification_prompt,
        normalized_documents=normalized_docs,
        execution_trace=ExecutionTrace(
            session_id=final_state.session_id,
            total_duration_ms=round(total_duration_ms, 2),
            steps=final_state.execution_trace,
        ),
        warnings=all_warnings,
        errors=final_state.errors,
        evidence=evidence,
        cross_source_analysis=cross_source_analysis,
    )
