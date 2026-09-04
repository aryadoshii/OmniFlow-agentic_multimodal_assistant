"""Abstract base interface for deterministic tools."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Minimal interface for deterministic Python tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief summary of the tool's purpose and inputs."""
        pass

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Executes the tool logic with keyword arguments."""
        pass
