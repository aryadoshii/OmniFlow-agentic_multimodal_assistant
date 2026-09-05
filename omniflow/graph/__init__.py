"""Agent orchestration graph package for OmniFlow.

Owns workflow control flow only (LangGraph state graph construction and
routing) -- it does not own tool execution (ToolRegistry), retrieval
(RAGService), or LLM access (GeminiProvider). Nodes call into those
abstractions in later phases; this package never implements their
responsibilities itself.
"""

from omniflow.graph.builder import build_graph, run_graph

__all__ = ["build_graph", "run_graph"]
