"""Centralized configuration management for OmniFlow using Pydantic Settings."""

from functools import lru_cache
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
