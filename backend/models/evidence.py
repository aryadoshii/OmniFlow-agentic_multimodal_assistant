"""Evidence/provenance models -- answer provenance as a first-class citizen.

An ``EvidenceReference`` identifies WHERE a piece of the final answer's
grounding came from (a RAG chunk, a directly-ingested document, or a
YouTube transcript tool result). These are built strictly from data the
existing pipeline already produced (see backend/agents/provenance.py) --
never invented, and never containing prompts or chain-of-thought.
"""

from typing import Literal
from pydantic import BaseModel, Field


class EvidenceReference(BaseModel):
    """One piece of source-attributed evidence backing (part of) the answer.

    Fields that cannot be honestly determined for a given source are left
    ``None`` rather than guessed -- e.g. ``page`` is only ever populated
    when a chunk can be unambiguously attributed to a single PDF page (see
    provenance.py's docstring for why multi-page attribution is not
    currently possible), and ``segment_start_seconds``/``segment_end_seconds``
    are only populated when the source actually carries timing data
    (YouTube transcripts; uploaded audio files do not -- faster-whisper's
    per-segment timestamps are not preserved past transcription).
    """

    source: Literal["rag", "document", "youtube_transcript"] = Field(
        ..., description="Which part of the pipeline this evidence came from."
    )
    document_id: str | None = Field(
        default=None, description="Originating NormalizedDocument id, when applicable."
    )
    filename: str | None = Field(
        default=None, description="Originating filename or source URL, when applicable."
    )
    source_type: str | None = Field(
        default=None,
        description="Modality of the source (pdf, text, image, audio, youtube), when applicable.",
    )
    extraction_method: str | None = Field(
        default=None, description="Extraction method used on the source, when applicable."
    )
    page: int | None = Field(
        default=None,
        description="PDF page number, only when unambiguous (see provenance.py) -- "
        "never guessed for a multi-page document.",
    )
    segment_start_seconds: float | None = Field(
        default=None, description="Segment start offset in seconds, when known."
    )
    segment_end_seconds: float | None = Field(
        default=None, description="Segment end offset in seconds, when known."
    )
    chunk_id: str | None = Field(
        default=None, description="RAG chunk id, when this evidence came from a RAG retrieval."
    )
    score: float | None = Field(
        default=None,
        description="RAG cosine similarity score (0.0-1.0), when this evidence came from RAG.",
    )
    excerpt: str = Field(
        ..., description="Short supporting excerpt from the actual source content -- never fabricated."
    )
