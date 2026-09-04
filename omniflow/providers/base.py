"""Abstract base interface for LLM providers."""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Minimal interface for LLM provider abstractions."""

    @abstractmethod
    def generate(
        self, prompt: str, system_instruction: str | None = None
    ) -> str:
        """Generates a text completion given a prompt and optional system instruction."""
        pass
