"""RAG search tool for OmniFlow -- exposes RAGService through the deterministic Tool Framework.

Wraps RAGService.retrieve() so a future orchestrator can invoke semantic
search over previously indexed documents through the same ToolRegistry
interface as any other deterministic tool.

This tool NEVER generates an answer and NEVER calls an LLM. It returns
structured evidence (or an explicit "no evidence" result) for a future
synthesis step to consume -- exactly what RAGService itself already
produces (see omniflow/rag/service.py's RAGResult), wrapped to conform to
the Phase 3.1 BaseTool contract.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from omniflow.rag.service import RAGResult, RAGService, RetrievedChunk
from omniflow.tools.base import BaseTool

logger = logging.getLogger(__name__)


class RAGSearchInput(BaseModel):
    """Structured input for RAGSearchTool."""

    query: str = Field(
        ..., min_length=1, description="Free-text semantic search query."
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum candidate chunks to consider. Defaults to settings.rag_top_k.",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for a chunk to count as relevant evidence. Defaults to settings.rag_similarity_threshold.",
    )

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or blank.")
        return value


class RAGSearchOutput(BaseModel):
    """Structured evidence result from a RAG search -- never an LLM answer.

    ``status`` distinguishes the two non-exceptional outcomes of a search:
    evidence was found, or the corpus/query combination yielded nothing
    that cleared the similarity threshold. Invalid input (a blank query)
    and retrieval failures (embedding backend down, FAISS failure) are not
    represented here -- they surface as raised exceptions from
    ``ToolRegistry.execute()`` / ``RAGSearchTool.run()`` instead (see the
    Phase 3.6 report for the full four-way distinction).
    """

    query: str = Field(..., description="The original search query.")
    status: Literal["evidence_found", "no_evidence"] = Field(
        ..., description="Whether any evidence cleared the similarity threshold."
    )
    evidence: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Ranked, source-attributed evidence chunks, highest score first.",
    )
    evidence_count: int = Field(default=0, description="Number of evidence chunks returned.")
    top_k: int = Field(..., description="The top_k value actually used for this search.")
    score_threshold: float = Field(
        ..., description="The similarity threshold actually used for this search."
    )
    message: str | None = Field(
        default=None, description="Human-readable explanation when status is 'no_evidence'."
    )


class RAGSearchTool(BaseTool):
    """Deterministic semantic search over previously indexed documents.

    Thin, stateless wrapper around a caller-supplied RAGService instance --
    this tool holds no index state itself and performs no chunking or
    embedding on its own; it only validates input, delegates to
    RAGService.retrieve(), and reshapes the result into structured output.
    """

    def __init__(self, rag_service: RAGService) -> None:
        self._rag_service = rag_service

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Searches previously indexed documents for semantically relevant evidence "
            "matching a query. Requires 'query' (str). Optional 'top_k' (int) and "
            "'score_threshold' (float, 0.0-1.0), both defaulting to configured RAG "
            "settings if omitted. Returns structured, source-attributed evidence "
            "chunks or an explicit no-evidence result -- never an LLM-generated answer."
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return RAGSearchInput

    def run(self, tool_input: RAGSearchInput) -> RAGSearchOutput:
        """Delegates to RAGService.retrieve() and reshapes the result.

        Raises:
            InvalidInputError: Propagated from RAGService/EmbeddingService if
                                the query is effectively empty (defense in
                                depth -- RAGSearchInput's own validator
                                already rejects this at construction time).
            EmbeddingGenerationError: Propagated as-is if the embedding
                                      backend is unavailable or fails.
            RAGRetrievalError: Propagated as-is if FAISS search fails.
        """
        result: RAGResult = self._rag_service.retrieve(
            query=tool_input.query,
            top_k=tool_input.top_k,
            score_threshold=tool_input.score_threshold,
        )

        status: Literal["evidence_found", "no_evidence"] = (
            "evidence_found" if result.has_evidence else "no_evidence"
        )

        logger.debug(
            "RAGSearchTool: query(len=%d) -> status=%s, evidence_count=%d",
            len(tool_input.query),
            status,
            len(result.results),
        )

        return RAGSearchOutput(
            query=result.query,
            status=status,
            evidence=result.results,
            evidence_count=len(result.results),
            top_k=result.top_k,
            score_threshold=result.score_threshold,
            message=result.message,
        )
