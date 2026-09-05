"""Abstract base interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Minimal interface for LLM provider abstractions."""

    @abstractmethod
    def generate(
        self, prompt: str, system_instruction: str | None = None
    ) -> str:
        """Generates a text completion given a prompt and optional system instruction."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: type[ResponseModelT],
        system_instruction: str | None = None,
    ) -> ResponseModelT:
        """Generates a completion validated into the given Pydantic response model.

        Implementations must return a genuine instance of ``response_model``
        (constructed via that model's own validation), never raw text or an
        unvalidated dict -- callers should be able to trust the returned
        object without re-checking its shape.
        """
        pass
