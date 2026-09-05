"""Agent reasoning package for OmniFlow (Phase 4.3+).

Owns semantic intent understanding, structured planning, final-answer
synthesis, and deterministic output validation. It performs no tool
execution and no direct RAG/FAISS/YouTube/processor calls -- those remain
the responsibility of ToolRegistry and its registered tools, invoked by the
graph's execute_tool node, not by this package.
"""

from omniflow.agents.intent import IntentResult, IntentType, understand_intent
from omniflow.agents.planner import Plan, PlanStep, create_plan
from omniflow.agents.reference_resolver import (
    DocumentCatalogEntry,
    build_document_catalog,
    resolve_unambiguous_references,
)
from omniflow.agents.synthesizer import synthesize_answer
from omniflow.agents.validator import ValidationResult, validate_structure

__all__ = [
    "DocumentCatalogEntry",
    "IntentResult",
    "IntentType",
    "Plan",
    "PlanStep",
    "ValidationResult",
    "build_document_catalog",
    "create_plan",
    "resolve_unambiguous_references",
    "synthesize_answer",
    "understand_intent",
    "validate_structure",
]
