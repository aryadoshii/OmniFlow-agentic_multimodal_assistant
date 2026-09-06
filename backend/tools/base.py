"""Abstract base interface for deterministic OmniFlow tools.

A tool is a synchronous, deterministic unit of work with a declared
structured input model and a structured output model. Tools are plain
Python components wrapped in a common, introspectable interface. They
have no awareness of LLMs, prompts, planning, or orchestration -- that
boundary is owned by future orchestration layers, not by tools themselves.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel


class BaseTool(ABC):
    """Minimal interface for deterministic tools with structured I/O."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable, unique identifier for this tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief human-readable summary of the tool's purpose and inputs."""
        pass

    @property
    @abstractmethod
    def input_model(self) -> type[BaseModel]:
        """Pydantic model class describing this tool's expected structured input."""
        pass

    @abstractmethod
    def run(self, tool_input: BaseModel) -> BaseModel:
        """Executes the tool deterministically given validated structured input.

        Implementations must return a ``BaseModel`` instance representing
        structured output, and should raise ``ToolExecutionError`` for
        internal failures rather than returning error sentinels.
        """
        pass
