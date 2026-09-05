"""Agent-facing API route: the final, high-level entry point into OmniFlow.

Unlike /ingest (a lower-level endpoint that only normalizes inputs), this
route accepts a user query plus optional files, ingests the files through
the existing ingestion layer, builds a fresh AgentState, executes the full
LangGraph workflow (intent -> clarity check -> plan/execute/observe/replan
-> synthesize -> validate), and returns the structured OmniFlowResponse.

Every request gets its own fresh RAGService/ToolRegistry -- no index or
tool state persists across requests (no persistent memory yet, per Phase 4
scope). This route owns no orchestration logic itself; it only ingests,
constructs state, calls run_graph(), and reshapes the result.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from functools import lru_cache

from omniflow.api.routes.ingest import get_ingestion_service
from omniflow.exceptions import InvalidInputError
from omniflow.graph import run_graph
from omniflow.models.request import UploadedInput
from omniflow.models.response import OmniFlowResponse
from omniflow.models.state import AgentState
from omniflow.models.trace import ExecutionTrace
from omniflow.providers.base import BaseLLMProvider
from omniflow.providers.gemini_provider import GeminiProvider
from omniflow.rag.embeddings import EmbeddingService
from omniflow.rag.service import RAGService
from omniflow.services.ingestion import IngestionService
from omniflow.tools.rag_search import RAGSearchTool
from omniflow.tools.registry import ToolRegistry
from omniflow.tools.youtube import YouTubeTranscriptTool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agent"])


def get_llm_provider() -> BaseLLMProvider:
    """Dependency provider for the BaseLLMProvider used by the workflow.

    A fresh GeminiProvider per request (lightweight, no network call at
    construction -- see GeminiProvider's own docstring). Tests override
    this dependency with a mock provider; no real Gemini call is made in
    the test suite.
    """
    return GeminiProvider()


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


def get_tool_registry(rag_service: RAGService = Depends(get_rag_service)) -> ToolRegistry:
    """Dependency provider for the ToolRegistry the graph selects tools from.

    Bundles every tool the planner is allowed to select: retrieval
    (bound to this request's own RAGService) and YouTube transcript
    retrieval (stateless).
    """
    registry = ToolRegistry()
    registry.register(RAGSearchTool(rag_service))
    registry.register(YouTubeTranscriptTool())
    return registry


@router.post("/query", response_model=OmniFlowResponse)
async def query_agent(
    query: Annotated[str, Form(description="The user's question or instruction.")],
    session_id: Annotated[
        str | None, Form(description="Optional session or conversation identifier.")
    ] = None,
    files: Annotated[
        list[UploadFile] | None,
        File(description="Optional list of file uploads (PDF, image, audio)."),
    ] = None,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
    rag_service: RAGService = Depends(get_rag_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
) -> OmniFlowResponse:
    """Runs the full agent workflow for a user query plus optional files.

    Ingests any uploaded files via the existing IngestionService, indexes
    them into a fresh per-request RAGService, constructs an AgentState, and
    executes the LangGraph workflow end to end -- intent understanding,
    ambiguity check (and, if ambiguous, an early stop with a clarifying
    question), bounded plan/execute/observe/replan, synthesis, and
    structural output validation.
    """
    if not query or not query.strip():
        raise InvalidInputError("query must not be empty.")

    uploaded_inputs: list[UploadedInput] = []
    file_payloads: list[tuple[bytes, str, str | None]] = []

    if files:
        for file in files:
            filename = file.filename or "upload"
            content = await file.read()
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

    if normalized_docs:
        rag_service.index_documents(normalized_docs)

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
    )
