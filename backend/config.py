"""Centralized configuration management for OmniFlow using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the project root (this file's parent's parent) rather than a
# bare ".env" -- a relative path is resolved against the process's current
# working directory, so it silently finds nothing (pydantic-settings does
# not error on a missing env file) whenever uvicorn is launched from
# anywhere other than the exact repo root. An absolute path makes local
# .env loading independent of where the process is started from.
_PROJECT_ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Metadata
    app_name: str = "OmniFlow"
    app_version: str = "0.1.0"
    app_env: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        alias="APP_ENV",
    )

    # Server Settings
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # LLM Provider Configuration
    gemini_api_key: SecretStr | None = Field(
        default=None,
        alias="GEMINI_API_KEY",
        description="Google Gemini API key, optional in Phase 1 foundation.",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        alias="LLM_MODEL",
        description="Default Gemini model to use in future phases.",
    )
    gemini_timeout_seconds: float = Field(
        default=30.0,
        alias="GEMINI_TIMEOUT_SECONDS",
        gt=0,
        description="Request timeout in seconds for Gemini API calls.",
    )
    gemini_max_retries: int = Field(
        default=2,
        alias="GEMINI_MAX_RETRIES",
        ge=0,
        description="Maximum number of retries GeminiProvider performs after an initial "
        "failed request, for transient errors only (5xx, 429, timeouts, network errors). "
        "0 disables retries entirely.",
    )
    gemini_retry_backoff_seconds: float = Field(
        default=1.0,
        alias="GEMINI_RETRY_BACKOFF_SECONDS",
        ge=0,
        description="Base delay in seconds for GeminiProvider's exponential backoff "
        "between retries (delay = backoff * (2 ** attempt)).",
    )

    # Upload and Ingestion Limits
    max_upload_size_mb: int = Field(
        default=25,
        alias="MAX_UPLOAD_SIZE_MB",
        ge=1,
        le=100,
        description="Maximum allowed upload payload size in megabytes.",
    )

    # OCR Configuration
    ocr_language: str = Field(
        default="eng",
        alias="OCR_LANGUAGE",
        description="Default language code for Tesseract OCR.",
    )
    tesseract_cmd: str | None = Field(
        default=None,
        alias="TESSERACT_CMD",
        description="Optional explicit path to the tesseract executable.",
    )
    pdf_native_text_char_threshold: int = Field(
        default=30,
        alias="PDF_NATIVE_TEXT_CHAR_THRESHOLD",
        ge=0,
        description="Minimum characters required to consider a PDF page native text rather than scanned.",
    )

    # Speech-to-Text (faster-whisper) Configuration
    whisper_model_size: str = Field(
        default="tiny",
        alias="WHISPER_MODEL_SIZE",
        description="Whisper model size (tiny, base, small, medium, large). Default is lightweight tiny for CPU.",
    )
    whisper_device: str = Field(
        default="cpu",
        alias="WHISPER_DEVICE",
        description="Device to run faster-whisper on (cpu or cuda).",
    )
    whisper_compute_type: str = Field(
        default="int8",
        alias="WHISPER_COMPUTE_TYPE",
        description="Quantization/compute type for faster-whisper (int8, float32, float16).",
    )

    # Embedding Configuration (Phase 3)
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
        description="Sentence-transformers model for local CPU embeddings.",
    )
    embedding_device: str = Field(
        default="cpu",
        alias="EMBEDDING_DEVICE",
        description="Device for embedding inference (cpu or cuda).",
    )
    embedding_batch_size: int = Field(
        default=32,
        alias="EMBEDDING_BATCH_SIZE",
        ge=1,
        description="Batch size used when encoding multiple texts in a single embedding call.",
    )

    # YouTube Transcript Tool Configuration (Phase 3.2)
    youtube_transcript_max_chars: int = Field(
        default=200_000,
        alias="YOUTUBE_TRANSCRIPT_MAX_CHARS",
        ge=1000,
        description="Maximum characters retained from a single YouTube transcript, to bound memory usage on very long videos.",
    )

    # RAG Chunking and Retrieval Configuration (Phase 3)
    rag_chunk_size: int = Field(
        default=500,
        alias="RAG_CHUNK_SIZE",
        ge=50,
        description="Target character count per document chunk.",
    )
    rag_chunk_overlap: int = Field(
        default=50,
        alias="RAG_CHUNK_OVERLAP",
        ge=0,
        description="Character overlap between adjacent chunks.",
    )
    rag_top_k: int = Field(
        default=4,
        alias="RAG_TOP_K",
        ge=1,
        description="Number of top similar chunks returned per retrieval query.",
    )
    rag_similarity_threshold: float = Field(
        default=0.2,
        alias="RAG_SIMILARITY_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score required for a chunk to be returned.",
    )

    # Bounded Agentic Execution (Phase 4.5)
    max_agent_steps: int = Field(
        default=6,
        alias="MAX_AGENT_STEPS",
        ge=1,
        description="Maximum number of plan/execute/observe cycles allowed in a single "
        "workflow run, regardless of what the planner decides -- the hard ceiling that "
        "guarantees the agent loop always terminates.",
    )
    max_tool_calls: int = Field(
        default=6,
        alias="MAX_TOOL_CALLS",
        ge=1,
        description="Maximum total tool invocations allowed in a single workflow run.",
    )
    max_retries: int = Field(
        default=2,
        alias="MAX_RETRIES",
        ge=1,
        description="Maximum times the same tool (by name) may be invoked within a single "
        "workflow run. Prevents the replanning loop from calling the same tool endlessly "
        "(e.g. repeating a RAG query that already returned no evidence).",
    )

    # Conversation History (SQLite, see backend/services/history_store.py)
    history_db_path: str = Field(
        default="database/omniflow_history.db",
        alias="HISTORY_DB_PATH",
        description="Path to the SQLite conversation-history database file. Relative "
        "paths resolve against the project root; override with an absolute path if "
        "a deployment target needs the file to live elsewhere (e.g. a mounted disk).",
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance to avoid repeated disk reads."""
    return Settings()
