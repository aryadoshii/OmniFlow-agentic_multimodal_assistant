"""Deterministic document chunking for RAG ingestion in OmniFlow."""

import uuid
from typing import Any
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.models.document import NormalizedDocument


class DocumentChunk(BaseModel):
    """Represents a single text chunk extracted from a NormalizedDocument."""

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this chunk.",
    )
    document_id: str = Field(
        ..., description="ID of the originating NormalizedDocument."
    )
    filename: str = Field(
        ..., description="Original filename of the source document."
    )
    source_type: str = Field(
        ..., description="Modality of the source (text, pdf, image, audio)."
    )
    extraction_method: str = Field(
        ..., description="Extraction method used on the source document (native_text, ocr, mixed, speech_to_text, direct_input)."
    )
    content: str = Field(..., description="The text content of this chunk.")
    chunk_index: int = Field(
        ..., description="Zero-based sequential position of this chunk within the document."
    )
    total_chunks: int = Field(
        ..., description="Total number of chunks produced for the originating document."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata inherited from the source document.",
    )


class DocumentChunker:
    """Splits NormalizedDocument content into overlapping chunks using recursive splitting.

    Hierarchy of split delimiters (tried in order):
        1. Paragraph breaks (``\\n\\n``)
        2. Single newlines (``\\n``)
        3. Sentence boundaries (``". "``)
        4. Whitespace (`` ``)
        5. Hard character split (no delimiter)

    ``chunk_size`` is the normal target/maximum chunk length in characters,
    with one deliberate exception: if the final chunk produced for a document
    would be a pathologically small trailing fragment (shorter than
    ``_MIN_CHUNK_RATIO`` of ``chunk_size``), it is merged into the preceding
    chunk instead of being left as its own, near-useless standalone chunk
    (see ``_merge_pathological_tail``). That merged chunk may therefore
    modestly exceed ``chunk_size``. This is intentional trailing-fragment
    handling, not a bug -- ``chunk_size`` is not a hard, universally-enforced
    maximum.
    """

    _DELIMITERS = ["\n\n", "\n", ". ", " ", ""]

    # A trailing chunk shorter than this fraction of chunk_size is merged into
    # its predecessor rather than left as a pathological tiny fragment.
    _MIN_CHUNK_RATIO = 0.2

    def __init__(
        self, chunk_size: int | None = None, chunk_overlap: int | None = None
    ) -> None:
        settings = get_settings()
        chunk_size = chunk_size if chunk_size is not None else settings.rag_chunk_size
        chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.rag_chunk_overlap
        )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document: NormalizedDocument) -> list[DocumentChunk]:
        """Splits a single NormalizedDocument into overlapping DocumentChunks.

        Returns an empty list for documents with empty or whitespace-only content.
        """
        text = document.content.strip()
        if not text:
            return []

        raw_chunks = self._split_text(text)
        total = len(raw_chunks)

        return [
            DocumentChunk(
                document_id=document.id,
                filename=document.filename,
                source_type=document.source_type.value,
                extraction_method=document.extraction_method.value,
                content=chunk,
                chunk_index=i,
                total_chunks=total,
                metadata={**document.metadata},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def chunk_documents(
        self, documents: list[NormalizedDocument]
    ) -> list[DocumentChunk]:
        """Chunks multiple NormalizedDocuments, preserving document order."""
        result: list[DocumentChunk] = []
        for doc in documents:
            result.extend(self.chunk_document(doc))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """Applies recursive delimiter splitting then merges pieces into sized chunks."""
        pieces = self._recursive_split(text, self._DELIMITERS)
        return self._merge_pieces(pieces)

    def _recursive_split(self, text: str, delimiters: list[str]) -> list[str]:
        """Splits text on the first delimiter that produces more than one piece.

        Falls back to the next delimiter when a split produces only a single piece
        that still exceeds chunk_size.
        """
        if not delimiters:
            # Hard character split as last resort
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
                if text[i : i + self.chunk_size]
            ]

        delimiter = delimiters[0]
        remaining_delimiters = delimiters[1:]

        if delimiter == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
                if text[i : i + self.chunk_size]
            ]

        parts = text.split(delimiter)
        if len(parts) == 1:
            # This delimiter didn't help; try next
            return self._recursive_split(text, remaining_delimiters)

        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > self.chunk_size:
                # Sub-split the oversized piece using finer delimiters
                result.extend(self._recursive_split(part, remaining_delimiters))
            else:
                result.append(part)
        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Merges small pieces into chunks of at most chunk_size characters with overlap."""
        if not pieces:
            return []

        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = (current + " " + piece).strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Start new chunk with overlap from previous chunk
                if self.chunk_overlap > 0 and current:
                    overlap_text = current[-self.chunk_overlap:]
                    current = (overlap_text + " " + piece).strip()
                else:
                    current = piece

        if current:
            chunks.append(current)

        return self._merge_pathological_tail(chunks)

    def _merge_pathological_tail(self, chunks: list[str]) -> list[str]:
        """Merges a pathologically small trailing chunk into its predecessor.

        A final chunk shorter than ``_MIN_CHUNK_RATIO * chunk_size`` carries
        little standalone retrieval value and is folded into the previous
        chunk instead, at the cost of that chunk modestly exceeding
        ``chunk_size``. Only ever merges the single trailing chunk; earlier
        chunks are left as produced.
        """
        if len(chunks) <= 1:
            return chunks

        min_len = max(1, int(self.chunk_size * self._MIN_CHUNK_RATIO))
        if len(chunks[-1]) < min_len:
            merged_tail = (chunks[-2] + " " + chunks[-1]).strip()
            return chunks[:-2] + [merged_tail]

        return chunks
