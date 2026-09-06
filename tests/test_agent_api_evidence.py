"""Integration tests for the /query response's new `evidence` and
`cross_source_analysis` fields (answer provenance + cross-modal analysis).

Same fully-mocked-provider pattern as test_agent_api.py -- no real Gemini
call, embedding model, or FAISS index is touched.
"""

import json

from fastapi.testclient import TestClient

from backend.agents.cross_source import CrossSourceAnalysis, SourceUnderstanding
from backend.agents.intent import IntentResult, IntentType
from backend.agents.planner import Plan, _PlanSchema, _PlanStepSchema
from backend.api.routes.agent import get_llm_provider, get_rag_service
from backend.providers.base import BaseLLMProvider
from unittest.mock import MagicMock


def _plan_to_wire_schema(plan: Plan) -> _PlanSchema:
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
    """Extends the other test files' double with CrossSourceAnalysis support."""

    def __init__(self, structured_outputs: dict, text_output: str = "A synthesized answer.") -> None:
        self._structured_outputs = structured_outputs
        self._text_output = text_output
        self._call_counts: dict = {}

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return self._text_output

    def generate_structured(self, prompt, response_model, system_instruction=None):
        lookup_model = Plan if response_model is _PlanSchema else response_model
        value = self._structured_outputs[lookup_model]
        if isinstance(value, list):
            idx = self._call_counts.get(lookup_model, 0)
            self._call_counts[lookup_model] = idx + 1
            value = value[min(idx, len(value) - 1)]
        if lookup_model is Plan and response_model is _PlanSchema:
            return _plan_to_wire_schema(value)
        return value


def _intent_result(**overrides) -> IntentResult:
    defaults = dict(
        intent=IntentType.GENERAL_CONVERSATION,
        constraints=[],
        referenced_inputs=[],
        relevant_references=[],
        is_ambiguous=False,
        needs_clarification=False,
        clarification_question=None,
        explanation="Answer directly.",
    )
    defaults.update(overrides)
    return IntentResult(**defaults)


def _override_agent_dependencies(client: TestClient, provider: BaseLLMProvider) -> MagicMock:
    rag_service_mock = MagicMock()
    client.app.dependency_overrides[get_llm_provider] = lambda: provider
    client.app.dependency_overrides[get_rag_service] = lambda: rag_service_mock
    return rag_service_mock


class TestEvidenceField:
    def test_direct_context_answer_includes_document_evidence(self, client: TestClient) -> None:
        provider = _FakeLLMProvider(
            {IntentResult: _intent_result(intent=IntentType.SUMMARIZATION), Plan: Plan(steps=[])}
        )
        _override_agent_dependencies(client, provider)

        response = client.post(
            "/query",
            data={"query": "Summarize this file."},
            files=[("files", ("notes.txt", b"Quarterly revenue increased.", "text/plain"))],
        )

        assert response.status_code == 200
        evidence = response.json()["evidence"]
        assert len(evidence) == 1
        assert evidence[0]["source"] == "document"
        assert evidence[0]["filename"] == "notes.txt"
        assert "Quarterly revenue" in evidence[0]["excerpt"]

    def test_general_conversation_with_no_files_has_no_evidence(self, client: TestClient) -> None:
        provider = _FakeLLMProvider({IntentResult: _intent_result(), Plan: Plan(steps=[])})
        _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "Hi there."})

        assert response.status_code == 200
        assert response.json()["evidence"] == []


class TestCrossSourceAnalysisField:
    def test_comparison_intent_with_two_files_populates_analysis(self, client: TestClient) -> None:
        analysis = CrossSourceAnalysis(
            relationship="partial_overlap",
            sources=[
                SourceUnderstanding(document_id="d1", filename="a.txt", topic="Validation", evidence="Discusses validation."),
                SourceUnderstanding(document_id="d2", filename="b.txt", topic="Workflow", evidence="Discusses workflow."),
            ],
            shared_concepts=["Clinical research"],
            differences=["a.txt is about validation; b.txt is about workflow."],
            explanation="Related but not the same specific topic.",
        )
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.COMPARISON),
                Plan: Plan(steps=[]),
                CrossSourceAnalysis: analysis,
            }
        )
        _override_agent_dependencies(client, provider)

        response = client.post(
            "/query",
            data={"query": "Do these two files discuss the same topic?"},
            files=[
                ("files", ("a.txt", b"Validation content.", "text/plain")),
                ("files", ("b.txt", b"Workflow content.", "text/plain")),
            ],
        )

        assert response.status_code == 200
        data = response.json()["cross_source_analysis"]
        assert data is not None
        assert data["relationship"] == "partial_overlap"
        assert len(data["sources"]) == 2
        assert data["shared_concepts"] == ["Clinical research"]

    def test_comparison_intent_with_only_one_file_is_none(self, client: TestClient) -> None:
        provider = _FakeLLMProvider(
            {IntentResult: _intent_result(intent=IntentType.COMPARISON), Plan: Plan(steps=[])}
        )
        _override_agent_dependencies(client, provider)

        response = client.post(
            "/query",
            data={"query": "Compare this."},
            files=[("files", ("a.txt", b"Some content.", "text/plain"))],
        )

        assert response.status_code == 200
        assert response.json()["cross_source_analysis"] is None

    def test_non_comparison_intent_with_two_files_is_none(self, client: TestClient) -> None:
        """Even with 2+ files, cross-source analysis must not run unless the
        classified intent is actually COMPARISON -- it must never fire on
        every multi-file request regardless of what was asked."""
        provider = _FakeLLMProvider(
            {IntentResult: _intent_result(intent=IntentType.SUMMARIZATION), Plan: Plan(steps=[])}
        )
        _override_agent_dependencies(client, provider)

        response = client.post(
            "/query",
            data={"query": "Summarize both files."},
            files=[
                ("files", ("a.txt", b"Some content.", "text/plain")),
                ("files", ("b.txt", b"Other content.", "text/plain")),
            ],
        )

        assert response.status_code == 200
        assert response.json()["cross_source_analysis"] is None
