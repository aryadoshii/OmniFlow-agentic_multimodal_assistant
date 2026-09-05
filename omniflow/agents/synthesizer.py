"""Final response synthesis (Phase 4.7).

Generates the user-facing ``final_answer`` from everything the workflow has
gathered so far: the original request, classified intent/constraints,
aggregated document context, and any tool results. This is the only place
in the workflow that produces free-text output for the user -- it never
executes a tool and never calls RAGService/FAISS/YouTube/the Gemini SDK
directly, only ``BaseLLMProvider``.

Synthesis is deliberately a single ``BaseLLMProvider.generate()`` call
(plain text, not structured output) -- the final answer is free-form prose,
not a typed object. Grounding/no-fabrication discipline is enforced through
the system instruction plus deterministic post-hoc structural validation
(see ``omniflow.agents.validator``), not through a response schema.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from omniflow.providers.base import BaseLLMProvider

if TYPE_CHECKING:
    # Deferred to break a circular import -- see reference_resolver.py for why.
    from omniflow.models.state import AgentState

logger = logging.getLogger(__name__)

# Internal prompt-construction bounds, not user-facing tunables -- deliberately
# not added to Settings (mirrors omniflow.agents.intent's _MAX_DOCUMENT_PREVIEW_CHARS).
_MAX_CONTEXT_CHARS = 12000
_MAX_TOOL_RESULTS_CHARS = 6000

_SYNTHESIS_SYSTEM_INSTRUCTION = """\
You are the final response-synthesis component of a deterministic \
multimodal assistant. Produce the single best final answer for the user, \
in plain text -- never JSON, and never a dump of internal state or tool names.

Grounding rules:
- Base your answer ONLY on the provided request, document context, and \
tool results. Never invent facts, documents, or evidence that were not provided.
- If a retrieval/search result reports no evidence found (e.g. a \
"no_evidence" status or an empty evidence/results list), say so plainly \
for that part of the request instead of fabricating an answer.
- If the provided context and tool results are insufficient to fully \
answer, answer what you can and clearly state what could not be determined.

Response forms (choose whichever the intent/request calls for):
- Direct conversational answers for general conversation.
- Concise summaries for summarization requests.
- Structured extraction (e.g. lists of action items) for extraction requests.
- Sentiment analysis: state a clear sentiment label (e.g. positive/negative/
  neutral/mixed), a confidence level for that label, and a brief
  justification grounded in the text.
- Code explanation: identify the programming language, explain what the
  code does, point out any bugs or issues you can identify, and give a
  brief complexity assessment (time/space) when it applies.
- Transcription/audio summaries: summarize the transcript, and report the
  audio's duration if it is present in the provided document metadata.
- Comparisons: address all requested sources side by side, not just one.

Constraint rules:
- Honor every listed constraint exactly (e.g. an exact bullet count, a \
one-line summary, a target audience) -- these came directly from the \
user's own wording.
- Do not include chain-of-thought, meta-commentary about your own \
process, or references to internal tool/step names.
"""

_CORRECTION_SYSTEM_INSTRUCTION_SUFFIX = """

CORRECTION PASS: Your previous answer violated one or more structural \
constraints, listed below. Revise the answer to satisfy them while \
preserving everything that was already correct and evidence-grounded. \
Return only the corrected final answer, not a description of the changes.
"""


def _serialize_tool_results(tool_results: dict[str, Any]) -> str:
    """Renders tool_results as bounded, safe JSON for the synthesis prompt."""
    if not tool_results:
        return "(no tool results)"

    safe: dict[str, Any] = {
        key: (value.model_dump(mode="json") if isinstance(value, BaseModel) else value)
        for key, value in tool_results.items()
    }
    rendered = json.dumps(safe, default=str)
    if len(rendered) > _MAX_TOOL_RESULTS_CHARS:
        rendered = rendered[:_MAX_TOOL_RESULTS_CHARS] + "... (truncated)"
    return rendered


def _build_synthesis_prompt(state: AgentState, correction_feedback: str | None) -> str:
    intent_value = state.intent_result.intent.value if state.intent_result else state.detected_intent
    constraints = state.intent_result.constraints if state.intent_result else state.constraints
    context = state.unified_context[:_MAX_CONTEXT_CHARS] if state.unified_context else "(no document context)"

    sections = [
        f"User request:\n{state.original_request}",
        f"Classified intent: {intent_value or 'unknown'}",
        f"Constraints: {json.dumps(constraints)}",
        f"Normalized document context:\n{context}",
        f"Tool results so far:\n{_serialize_tool_results(state.tool_results)}",
    ]

    if state.errors:
        sections.append(
            "Errors encountered during execution (be honest about any impact on "
            f"completeness -- do not hide them from the user): {json.dumps(state.errors)}"
        )

    if correction_feedback:
        sections.append(f"Previous answer:\n{state.final_answer or ''}")
        sections.append(f"Structural violations to fix: {correction_feedback}")

    return "\n\n".join(sections)


def synthesize_answer(
    state: AgentState,
    llm_provider: BaseLLMProvider,
    correction_feedback: str | None = None,
) -> str:
    """Generates the final free-text answer for the user.

    Args:
        state: The current request's AgentState. Only read, never mutated --
               callers (a future graph node) are responsible for merging the
               returned answer into a new AgentState.
        llm_provider: A BaseLLMProvider implementation (e.g. GeminiProvider).
                      This function never imports a concrete provider itself.
        correction_feedback: When set, a concise description of the
                              structural violations found in the PREVIOUS
                              synthesis attempt (see
                              omniflow.agents.validator.validate_structure)
                              -- triggers a bounded, single revision pass
                              rather than synthesizing from scratch.

    Returns:
        The synthesized answer text (plain text, never JSON), stripped of
        leading/trailing whitespace.

    Raises:
        ConfigurationError: Propagated as-is from llm_provider if
                             credentials/model configuration are invalid.
        ExternalProviderError: Propagated as-is from llm_provider if the LLM
                                request fails.
    """
    prompt = _build_synthesis_prompt(state, correction_feedback)
    system_instruction = _SYNTHESIS_SYSTEM_INSTRUCTION
    if correction_feedback:
        system_instruction += _CORRECTION_SYSTEM_INSTRUCTION_SUFFIX

    logger.info(
        "Synthesizing final answer (request_len=%d, tool_result_count=%d, correction=%s).",
        len(state.original_request),
        len(state.tool_results),
        bool(correction_feedback),
    )
    answer = llm_provider.generate(prompt, system_instruction=system_instruction)
    return answer.strip()
