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

from omniflow.models.document import NormalizedDocument
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

    Thin wrapper around a caller-supplied RAGService instance -- this tool
    performs no chunking or embedding logic of its own, only validates
    input, delegates to RAGService.index_documents()/retrieve(), and
    reshapes the result into structured output.

    Indexing is LAZY: the documents passed at construction are stored but
    NOT indexed until this tool's own first ``run()`` call, and at most
    once per instance thereafter (an ``_indexed`` flag guards repeated
    replan cycles from re-indexing the same documents). This matters
    because indexing is what forces the embedding backend (sentence-
    transformers/Torch) to load -- a request whose plan never selects
    rag_search (a direct-context answer) must never pay that cost just
    because files happened to be uploaded. See omniflow/api/routes/agent.py,
    which constructs one RAGSearchTool per request with that request's
    normalized_documents, whether or not the planner ends up using it.
    """

    def __init__(
        self,
        rag_service: RAGService,
        normalized_documents: list[NormalizedDocument] | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._documents = list(normalized_documents) if normalized_documents else []
        self._indexed = False

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
        """Indexes this tool's documents (once) then delegates to
        RAGService.retrieve(), reshaping the result.

        The first call indexes ``self._documents`` (an empty list is a
        cheap, explicit no-op -- see RAGService.index_documents()'s own
        docstring) before ever calling ``embed_query()``/``retrieve()``.
        Every subsequent call on this same instance skips indexing
        entirely, so a replanning loop that calls rag_search more than
        once for one request indexes its documents exactly once.

        Raises:
            InvalidInputError: Propagated from RAGService/EmbeddingService if
                                the query is effectively empty (defense in
                                depth -- RAGSearchInput's own validator
                                already rejects this at construction time).
            EmbeddingGenerationError: Propagated as-is if the embedding
                                      backend is unavailable or fails
                                      (indexing or query embedding).
            RAGRetrievalError: Propagated as-is if FAISS indexing or search fails.
        """
        if not self._indexed:
            self._rag_service.index_documents(self._documents)
            self._indexed = True

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
