"""End-to-end integration tests for the OmniFlow agent workflow (Phase 4,
second half): tool execution, autonomous multi-step replanning, bounded
execution limits, clarification, synthesis, and output validation, all
exercised together through build_graph()/run_graph().

The LLM boundary (BaseLLMProvider) is fully mocked via a local
_FakeLLMProvider double -- no real Gemini call is ever made. The YouTube
boundary is mocked at YouTubeTranscriptApi (the same seam test_youtube_tool.py
uses). RAGService is mocked via MagicMock (the same seam test_rag_tool.py
uses) so no embedding model or FAISS index is touched. No network access
occurs anywhere in this file.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from omniflow.agents.intent import IntentResult, IntentType
from omniflow.agents.planner import Plan, PlanStep, _PlanSchema, _PlanStepSchema
from omniflow.config import Settings
from omniflow.graph import build_graph, run_graph
from omniflow.models.state import AgentState, WorkflowStatus
from omniflow.providers.base import BaseLLMProvider
from omniflow.rag.service import RAGResult, RetrievedChunk
from omniflow.tools.rag_search import RAGSearchTool
from omniflow.tools.registry import ToolRegistry
from omniflow.tools.youtube import YouTubeTranscriptTool

VALID_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------


def _plan_to_wire_schema(plan: Plan) -> _PlanSchema:
    """Converts a Plan into the Gemini-facing wire shape (inputs as a JSON
    string) a real GeminiProvider actually returns -- see planner.py's
    _PlanSchema docstring."""
    return _PlanSchema(
        steps=[
            _PlanStepSchema(
                step_id=step.step_id,
                tool_name=step.tool_name,
                purpose=step.purpose,
                inputs=json.dumps(step.inputs),
                expected_result=step.expected_result,
                depends_on=step.depends_on,
            )
            for step in plan.steps
        ]
    )


class _FakeLLMProvider(BaseLLMProvider):
    """Deterministic BaseLLMProvider test double driving the real graph.

    Structured outputs (IntentResult/Plan) are configured per response
    model, list values consumed one per call (repeating the last for extra
    calls). Text outputs (used only by synthesize) are configured the same
    way via `text_outputs`. Every prompt is recorded (keyed by response
    model, or under "generate" for plain-text calls) so tests can assert on
    what the graph actually fed the LLM -- e.g. that a replan's prompt
    reflects the previous tool's result.
    """

    def __init__(
        self,
        structured_outputs: dict,
        text_outputs: list[str] | str = "This is a synthesized test answer.",
    ) -> None:
        self._structured_outputs = structured_outputs
        self._text_outputs = text_outputs
        self._structured_call_counts: dict = {}
        self._text_call_count = 0
        self.received_prompts: dict = {"generate": []}

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.received_prompts["generate"].append(prompt)
        if isinstance(self._text_outputs, list):
            idx = self._text_call_count
            self._text_call_count += 1
            return self._text_outputs[min(idx, len(self._text_outputs) - 1)]
        return self._text_outputs

    def generate_structured(self, prompt, response_model, system_instruction=None):
        # The planner requests _PlanSchema (the Gemini-safe wire shape),
        # not Plan directly -- see planner.py's _PlanSchema docstring. Test
        # fixtures (including received_prompts assertions) key off the
        # plain Plan model, so translate here rather than in every test.
        lookup_model = Plan if response_model is _PlanSchema else response_model
        self.received_prompts.setdefault(lookup_model, []).append(prompt)
        value = self._structured_outputs[lookup_model]
        if isinstance(value, list):
            idx = self._structured_call_counts.get(lookup_model, 0)
            self._structured_call_counts[lookup_model] = idx + 1
            value = value[min(idx, len(value) - 1)]
        if lookup_model is Plan and response_model is _PlanSchema:
            return _plan_to_wire_schema(value)
        return value


@dataclass
class _FakeYouTubeSnippet:
    text: str
    start: float
    duration: float


class _FakeFetchedTranscript:
    def __init__(self, snippets: list[_FakeYouTubeSnippet]) -> None:
        self.snippets = snippets
        self.language = "English"
        self.language_code = "en"
        self.is_generated = False

    def __iter__(self):
        return iter(self.snippets)

    def __len__(self) -> int:
        return len(self.snippets)


def _mock_youtube_api_returning(transcript_text: str):
    mock_instance = MagicMock()
    mock_instance.fetch.return_value = _FakeFetchedTranscript(
        [_FakeYouTubeSnippet(text=transcript_text, start=0.0, duration=5.0)]
    )
    return MagicMock(return_value=mock_instance)


def _intent_result(**overrides) -> IntentResult:
    defaults = dict(
        intent=IntentType.QUESTION_ANSWERING,
        constraints=[],
        referenced_inputs=[],
        relevant_references=[],
        is_ambiguous=False,
        needs_clarification=False,
        clarification_question=None,
        explanation="Answer the request.",
    )
    defaults.update(overrides)
    return IntentResult(**defaults)


def _step(step_id: int = 0, tool_name: str | None = None, **overrides) -> PlanStep:
    defaults = dict(
        step_id=step_id,
        tool_name=tool_name,
        purpose="Do the thing.",
        inputs={},
        expected_result="The thing is done.",
        depends_on=None,
    )
    defaults.update(overrides)
    return PlanStep(**defaults)


def _rag_result(has_evidence: bool, content: str = "Revenue rose 15% in Q3.") -> RAGResult:
    if not has_evidence:
        return RAGResult(query="q", results=[], has_evidence=False, top_k=4, score_threshold=0.2, message="No evidence found.")
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        filename="report.pdf",
        source_type="pdf",
        extraction_method="native_text",
        content=content,
        score=0.9,
        chunk_index=0,
        total_chunks=1,
    )
    return RAGResult(query="q", results=[chunk], has_evidence=True, top_k=4, score_threshold=0.2)


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ---------------------------------------------------------------------------
# 1. Direct answer -- no tool needed
# ---------------------------------------------------------------------------


class TestDirectAnswer:
    def test_general_conversation_needs_no_tool(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.GENERAL_CONVERSATION),
                Plan: Plan(steps=[]),
            }
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(AgentState(original_request="Hello, how are you?"), compiled_graph=graph)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.final_answer == "This is a synthesized test answer."
        assert result.tool_call_history == []


# ---------------------------------------------------------------------------
# 2. RAG-backed question
# ---------------------------------------------------------------------------


class TestRAGBackedQuestion:
    def test_evidence_found_grounds_synthesis(self) -> None:
        rag_service = MagicMock()
        rag_service.retrieve.return_value = _rag_result(has_evidence=True)
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.QUESTION_ANSWERING),
                Plan: [
                    Plan(steps=[_step(0, tool_name="rag_search", inputs={"query": "Q3 revenue"})]),
                    Plan(steps=[]),
                ],
            }
        )
        graph = build_graph(_registry(RAGSearchTool(rag_service)), llm_provider=provider)
        result = run_graph(
            AgentState(original_request="What was Q3 revenue?"), compiled_graph=graph
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.tool_call_history == ["rag_search"]
        synthesis_prompt = provider.received_prompts["generate"][0]
        assert "evidence_found" in synthesis_prompt
        assert "Revenue rose 15%" in synthesis_prompt


# ---------------------------------------------------------------------------
# 3. YouTube workflow
# ---------------------------------------------------------------------------


class TestYouTubeWorkflow:
    def test_transcript_retrieved_and_grounds_synthesis(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.TRANSCRIPTION_SUMMARY),
                Plan: [
                    Plan(steps=[_step(0, tool_name="youtube_transcript", inputs={"url": VALID_YOUTUBE_URL})]),
                    Plan(steps=[]),
                ],
            }
        )
        graph = build_graph(_registry(YouTubeTranscriptTool()), llm_provider=provider)

        with patch("omniflow.tools.youtube.YouTubeTranscriptApi", _mock_youtube_api_returning("Welcome to the talk.")):
            result = run_graph(
                AgentState(original_request="Summarize this video."), compiled_graph=graph
            )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.tool_call_history == ["youtube_transcript"]
        synthesis_prompt = provider.received_prompts["generate"][0]
        assert "Welcome to the talk." in synthesis_prompt


# ---------------------------------------------------------------------------
# 4. Two-step workflow -- PDF discovers a YouTube reference
# ---------------------------------------------------------------------------


class TestTwoStepWorkflow:
    def test_youtube_then_rag_search_replan_depends_on_prior_result(self) -> None:
        """'Summarize the YouTube video mentioned in this PDF' -- the second
        planning call must be genuinely informed by the first tool's result,
        not a blindly pre-scripted static chain."""
        rag_service = MagicMock()
        rag_service.retrieve.return_value = _rag_result(has_evidence=True, content="Talk covered agentic AI.")
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.TRANSCRIPTION_SUMMARY),
                Plan: [
                    Plan(steps=[_step(0, tool_name="youtube_transcript", inputs={"url": VALID_YOUTUBE_URL})]),
                    Plan(steps=[_step(0, tool_name="rag_search", inputs={"query": "key topics"})]),
                    Plan(steps=[]),
                ],
            }
        )
        registry = _registry(YouTubeTranscriptTool(), RAGSearchTool(rag_service))
        graph = build_graph(registry, llm_provider=provider)

        with patch("omniflow.tools.youtube.YouTubeTranscriptApi", _mock_youtube_api_returning("Deep dive on agents.")):
            result = run_graph(
                AgentState(original_request="Summarize the YouTube video mentioned in this PDF."),
                compiled_graph=graph,
            )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.tool_call_history == ["youtube_transcript", "rag_search"]

        # The SECOND plan() call's prompt must include the first tool's
        # actual result -- proof the next action depends on real state,
        # not a static pre-scripted sequence.
        second_plan_prompt = provider.received_prompts[Plan][1]
        assert "Deep dive on agents." in second_plan_prompt


# ---------------------------------------------------------------------------
# 5. Multi-step workflow (3+ tool calls)
# ---------------------------------------------------------------------------


class TestMultiStepWorkflow:
    def test_three_sequential_tool_calls_before_completion(self) -> None:
        rag_service = MagicMock()
        rag_service.retrieve.return_value = _rag_result(has_evidence=True)
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.COMPARISON),
                Plan: [
                    Plan(steps=[_step(0, tool_name="youtube_transcript", inputs={"url": VALID_YOUTUBE_URL})]),
                    Plan(steps=[_step(0, tool_name="rag_search", inputs={"query": "first point"})]),
                    Plan(steps=[_step(0, tool_name="rag_search", inputs={"query": "second point"})]),
                    Plan(steps=[]),
                ],
            }
        )
        registry = _registry(YouTubeTranscriptTool(), RAGSearchTool(rag_service))
        graph = build_graph(registry, llm_provider=provider)

        with patch("omniflow.tools.youtube.YouTubeTranscriptApi", _mock_youtube_api_returning("transcript text")):
            result = run_graph(
                AgentState(original_request="Compare this audio with the PDF."), compiled_graph=graph
            )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.tool_call_history == ["youtube_transcript", "rag_search", "rag_search"]
        assert result.agent_step_count == 4


# ---------------------------------------------------------------------------
# 6/7. Ambiguous request -> clarification
# ---------------------------------------------------------------------------


class TestClarification:
    def test_ambiguous_request_stops_before_any_tool_invocation(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(
                    is_ambiguous=True,
                    needs_clarification=True,
                    clarification_question="Which document would you like summarized?",
                ),
                Plan: Plan(steps=[]),
            }
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(AgentState(original_request="Summarize it."), compiled_graph=graph)

        assert result.status == WorkflowStatus.AWAITING_CLARIFICATION
        assert result.clarification_prompt == "Which document would you like summarized?"
        assert result.tool_call_history == []
        step_names = [t.step_name for t in result.execution_trace]
        assert "node:plan" not in step_names
        assert "node:execute_tool" not in step_names
        assert "node:synthesize" not in step_names
        # No structured Plan call was ever made -- confirms the planner
        # itself was never invoked, not merely that its result was discarded.
        assert Plan not in provider.received_prompts

    def test_clarification_preserves_state_for_a_later_turn(self) -> None:
        """The state must retain enough information (original request,
        uploaded/normalized documents) that a future turn could continue
        the task once the user answers -- even though resuming isn't
        implemented yet (no persistent memory in this phase)."""
        from omniflow.models.document import ExtractionMethod, NormalizedDocument, SourceType

        doc = NormalizedDocument(
            filename="report.pdf",
            source_type=SourceType.PDF,
            mime_type="application/pdf",
            content="Quarterly figures.",
            extraction_method=ExtractionMethod.NATIVE_TEXT,
        )
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(
                    needs_clarification=True, clarification_question="Which section?"
                ),
                Plan: Plan(steps=[]),
            }
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(
            AgentState(original_request="Summarize the report.", normalized_documents=[doc]),
            compiled_graph=graph,
        )

        assert result.status == WorkflowStatus.AWAITING_CLARIFICATION
        assert result.original_request == "Summarize the report."
        assert len(result.normalized_documents) == 1
        assert result.normalized_documents[0].content == "Quarterly figures."


# ---------------------------------------------------------------------------
# 8. Tool failure
# ---------------------------------------------------------------------------


class TestToolFailure:
    def test_failed_tool_still_produces_a_best_effort_answer(self) -> None:
        rag_service = MagicMock()
        rag_service.retrieve.side_effect = RuntimeError("faiss exploded")
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(),
                Plan: Plan(steps=[_step(0, tool_name="rag_search", inputs={"query": "revenue"})]),
            }
        )
        graph = build_graph(_registry(RAGSearchTool(rag_service)), llm_provider=provider)
        result = run_graph(AgentState(original_request="What was revenue?"), compiled_graph=graph)

        assert result.status == WorkflowStatus.FAILED
        assert result.errors  # the failure is recorded, never silently swallowed
        # synthesize still runs (best-effort, honest about the failure)
        # rather than leaving the user with nothing.
        assert result.final_answer == "This is a synthesized test answer."
        synthesis_prompt = provider.received_prompts["generate"][0]
        assert "faiss exploded" in synthesis_prompt or "TOOL_EXECUTION_ERROR" in synthesis_prompt


# ---------------------------------------------------------------------------
# 9. No-evidence RAG
# ---------------------------------------------------------------------------


class TestNoEvidenceRAG:
    def test_no_evidence_completes_without_fabrication_signal_to_synthesis(self) -> None:
        rag_service = MagicMock()
        rag_service.retrieve.return_value = _rag_result(has_evidence=False)
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(),
                Plan: [
                    Plan(steps=[_step(0, tool_name="rag_search", inputs={"query": "unrelated topic"})]),
                    Plan(steps=[]),
                ],
            }
        )
        graph = build_graph(_registry(RAGSearchTool(rag_service)), llm_provider=provider)
        result = run_graph(AgentState(original_request="What about unrelated topic?"), compiled_graph=graph)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.errors == []  # no evidence is not an error
        synthesis_prompt = provider.received_prompts["generate"][0]
        assert "no_evidence" in synthesis_prompt


# ---------------------------------------------------------------------------
# 10. Execution budget
# ---------------------------------------------------------------------------


class TestExecutionBudget:
    def test_max_agent_steps_terminates_gracefully(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(),
                # Never returns an empty plan -- the budget, not the planner,
                # must be what stops this run.
                Plan: Plan(steps=[_step(0, tool_name="echo", inputs={"message": "hi"})]),
            }
        )
        from tests.test_graph import _EchoTool  # reuse the existing dummy tool

        graph = build_graph(_registry(_EchoTool()), llm_provider=provider)
        low_budget_settings = Settings(MAX_AGENT_STEPS=2, MAX_TOOL_CALLS=10, MAX_RETRIES=10)

        with patch("omniflow.graph.nodes.get_settings", return_value=low_budget_settings):
            result = run_graph(AgentState(original_request="loop forever"), compiled_graph=graph)

        assert result.status == WorkflowStatus.FAILED
        assert any("Maximum agent steps" in e for e in result.errors)
        assert len(result.tool_call_history) <= 2


# ---------------------------------------------------------------------------
# 11. Repeated-tool prevention
# ---------------------------------------------------------------------------


class TestRepeatedToolPrevention:
    def test_max_retries_stops_identical_repeated_tool_calls(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(),
                Plan: Plan(steps=[_step(0, tool_name="echo", inputs={"message": "hi"})]),
            }
        )
        from tests.test_graph import _EchoTool

        graph = build_graph(_registry(_EchoTool()), llm_provider=provider)
        generous_steps_settings = Settings(MAX_AGENT_STEPS=10, MAX_TOOL_CALLS=10, MAX_RETRIES=2)

        with patch("omniflow.graph.nodes.get_settings", return_value=generous_steps_settings):
            result = run_graph(AgentState(original_request="call echo forever"), compiled_graph=graph)

        assert result.status == WorkflowStatus.FAILED
        assert result.tool_call_history == ["echo", "echo"]  # stopped at max_retries, not before/after
        assert any("refusing to call it again" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Planner safety: only registered tools may ever be selected
# ---------------------------------------------------------------------------


class TestHallucinatedToolRejection:
    def test_plan_naming_an_unregistered_tool_fails_gracefully_not_a_crash(self) -> None:
        """Critical agent-behavior requirement: the planner must select
        only registered tools. create_plan() itself already validates this
        (see test_planner.py's unit-level coverage) -- this proves the
        FULL graph absorbs a hallucinated tool name as a controlled
        WorkflowStatus.FAILED outcome (still producing a best-effort
        answer) rather than letting run_graph() raise or hang."""
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.QUESTION_ANSWERING),
                # No tool named "made_up_tool" is registered below -- the
                # planner is hallucinating a capability that doesn't exist.
                Plan: Plan(steps=[_step(0, tool_name="made_up_tool", inputs={"query": "x"})]),
            }
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)  # empty registry

        result = run_graph(AgentState(original_request="Do something."), compiled_graph=graph)

        assert result.status == WorkflowStatus.FAILED
        assert any("made_up_tool" in e for e in result.errors)
        assert result.tool_call_history == []  # never actually invoked
        # Still reaches a best-effort answer rather than leaving the user
        # with nothing (see route_after_route_next: FAILED -> synthesize).
        assert result.final_answer == "This is a synthesized test answer."


class TestProviderFailure:
    def test_gemini_provider_failure_during_intent_understanding_fails_gracefully(self) -> None:
        """Distinct from 'no llm_provider configured' (ConfigurationError,
        covered elsewhere): this simulates a REAL, live Gemini call failing
        (rate limit, timeout, service outage) partway through a request --
        exactly the ExternalProviderError contract GeminiProvider itself
        raises (see test_gemini_provider.py). The graph must absorb it as
        a controlled FAILED outcome, never a raised exception or a hang."""
        from unittest.mock import MagicMock

        from omniflow.exceptions import ExternalProviderError

        failing_provider = MagicMock(spec=BaseLLMProvider)
        failing_provider.generate_structured.side_effect = ExternalProviderError(
            "Gemini rate limit or quota exceeded.", details={"reason": "rate_limited"}
        )
        failing_provider.generate.return_value = "Sorry, I could not complete this request."

        graph = build_graph(ToolRegistry(), llm_provider=failing_provider)
        result = run_graph(AgentState(original_request="Summarize this."), compiled_graph=graph)

        assert result.status == WorkflowStatus.FAILED
        assert any("EXTERNAL_PROVIDER_ERROR" in e for e in result.errors)
        # Still produces a best-effort final answer rather than nothing.
        assert result.final_answer == "Sorry, I could not complete this request."


# ---------------------------------------------------------------------------
# 12. Synthesis validation (bounded correction)
# ---------------------------------------------------------------------------


class TestAudioWorkflowEndToEnd:
    """Assignment workflow 1: Audio -> transcript + summaries + duration.

    Exercises the REAL ingestion pipeline (mocked only at the Whisper
    boundary, the same seam test_processors.py uses) feeding into the REAL
    prepare_context node, confirming a hardening fix: unified_context now
    carries document metadata (duration_seconds), not just transcript
    content -- previously duration could never have reached synthesis at
    all, since synthesize_answer() only ever reads unified_context.
    """

    def test_audio_duration_reaches_synthesis_prompt_via_real_ingestion(self) -> None:
        from omniflow.processors.audio_processor import AudioProcessor
        from omniflow.services.ingestion import IngestionService
        from omniflow.services.whisper_service import WhisperService
        from tests.test_processors import create_sample_wav_bytes

        mock_whisper = MagicMock(spec=WhisperService)
        mock_whisper.transcribe.return_value = (
            "Let's discuss the product launch schedule for Q3.",
            None,
            {"duration_seconds": 187.5, "language": "en"},
        )
        ingestion_service = IngestionService(processors=[AudioProcessor(whisper_service=mock_whisper)])
        wav_bytes = create_sample_wav_bytes(0.5)

        normalized_docs = ingestion_service.ingest_inputs(
            files=[(wav_bytes, "meeting.wav", "audio/wav")]
        )
        assert len(normalized_docs) == 1
        assert normalized_docs[0].metadata["duration_seconds"] == 187.5

        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.TRANSCRIPTION_SUMMARY),
                Plan: Plan(steps=[]),
            }
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(
            AgentState(
                original_request="Summarize this audio and tell me its duration.",
                normalized_documents=normalized_docs,
            ),
            compiled_graph=graph,
        )

        assert result.status == WorkflowStatus.COMPLETED
        synthesis_prompt = provider.received_prompts["generate"][0]
        assert "187.5" in synthesis_prompt
        assert "product launch schedule" in synthesis_prompt


class TestImageCodeWorkflowEndToEnd:
    """Assignment workflow 3: Image/code -> OCR + language + explanation +
    bugs + complexity. Exercises real ingestion (mocked only at the OCR
    boundary) and confirms the OCR'd code text actually reaches synthesis."""

    def test_ocr_extracted_code_reaches_synthesis_prompt_via_real_ingestion(self) -> None:
        from omniflow.processors.image_processor import ImageProcessor
        from omniflow.services.ingestion import IngestionService
        from omniflow.services.ocr_service import OCRService
        from tests.test_processors import create_sample_image_bytes

        mock_ocr = MagicMock(spec=OCRService)
        mock_ocr.extract_text_with_confidence.return_value = (
            "def divide(a, b):\n    return a / b",
            0.95,
            {"word_count": 5},
        )
        ingestion_service = IngestionService(processors=[ImageProcessor(ocr_service=mock_ocr)])
        img_bytes = create_sample_image_bytes("PNG")

        normalized_docs = ingestion_service.ingest_inputs(
            files=[(img_bytes, "snippet.png", "image/png")]
        )
        assert "def divide" in normalized_docs[0].content

        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.CODE_EXPLANATION),
                Plan: Plan(steps=[]),
            }
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(
            AgentState(
                original_request="Explain this code, list any bugs, and give its complexity.",
                normalized_documents=normalized_docs,
            ),
            compiled_graph=graph,
        )

        assert result.status == WorkflowStatus.COMPLETED
        synthesis_prompt = provider.received_prompts["generate"][0]
        assert "def divide" in synthesis_prompt


class TestSynthesisValidation:
    def test_structural_violation_triggers_one_bounded_correction(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(constraints=["exactly 2 bullets"]),
                Plan: Plan(steps=[]),
            },
            text_outputs=["- only one bullet", "- first bullet\n- second bullet"],
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(
            AgentState(original_request="List two things, exactly 2 bullets."), compiled_graph=graph
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.synthesis_attempts == 2
        assert result.final_answer == "- first bullet\n- second bullet"
        step_names = [t.step_name for t in result.execution_trace]
        assert step_names.count("node:synthesize") == 2
        assert step_names.count("node:validate_output") == 2
        # The second synthesize call must have received the violation as feedback.
        assert "bullet" in provider.received_prompts["generate"][1].lower()

    def test_still_invalid_after_correction_still_terminates(self) -> None:
        """The correction is bounded to exactly one retry -- even if still
        invalid, the run must complete rather than loop indefinitely."""
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(constraints=["exactly 5 bullets"]),
                Plan: Plan(steps=[]),
            },
            text_outputs="- still just one bullet",
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(AgentState(original_request="List five things."), compiled_graph=graph)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.synthesis_attempts == 2
        step_names = [t.step_name for t in result.execution_trace]
        assert step_names.count("node:synthesize") == 2

    def test_valid_output_on_first_attempt_never_triggers_correction(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(constraints=["exactly 1 bullets"]),
                Plan: Plan(steps=[]),
            },
            text_outputs="- exactly one bullet",
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(AgentState(original_request="List one thing."), compiled_graph=graph)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.synthesis_attempts == 1
        step_names = [t.step_name for t in result.execution_trace]
        assert step_names.count("node:synthesize") == 1
