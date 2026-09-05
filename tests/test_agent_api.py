"""Integration tests for the POST /query agent-facing API endpoint.

The LLM provider and RAGService are overridden via FastAPI dependency
overrides -- no real Gemini call, embedding model load, or FAISS index is
ever touched. /ingest's own ingestion behavior is already covered by
test_ingestion_api.py; these tests focus on the new endpoint's wiring:
request handling, graph execution, and response shaping.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from omniflow.agents.intent import IntentResult, IntentType
from omniflow.agents.planner import Plan, PlanStep
from omniflow.api.routes.agent import (
    get_embedding_service,
    get_llm_provider,
    get_rag_service,
)
from omniflow.providers.base import BaseLLMProvider
from omniflow.rag.embeddings import EmbeddingService


class _FakeLLMProvider(BaseLLMProvider):
    """Same minimal double used in test_agent_integration.py, kept local to
    this file so API tests stay self-contained."""

    def __init__(self, structured_outputs: dict, text_output: str = "A synthesized answer.") -> None:
        self._structured_outputs = structured_outputs
        self._text_output = text_output
        self._call_counts: dict = {}

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return self._text_output

    def generate_structured(self, prompt, response_model, system_instruction=None):
        value = self._structured_outputs[response_model]
        if isinstance(value, list):
            idx = self._call_counts.get(response_model, 0)
            self._call_counts[response_model] = idx + 1
            return value[min(idx, len(value) - 1)]
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
    """Overrides the LLM/RAG dependencies for one test and returns the SAME
    RAGService mock instance the route will actually use (so tests can
    assert on calls made to it), rather than a throwaway instance."""
    rag_service_mock = MagicMock()
    client.app.dependency_overrides[get_llm_provider] = lambda: provider
    client.app.dependency_overrides[get_rag_service] = lambda: rag_service_mock
    return rag_service_mock


class TestDirectAnswer:
    def test_query_without_files_returns_synthesized_answer(self, client: TestClient) -> None:
        provider = _FakeLLMProvider(
            {IntentResult: _intent_result(), Plan: Plan(steps=[])},
            text_output="Hello! I'm doing well, thanks for asking.",
        )
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "Hi, how are you?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Hello! I'm doing well, thanks for asking."
        assert data["clarification_needed"] is False
        assert data["clarification_prompt"] is None
        assert data["normalized_documents"] == []

    def test_response_includes_execution_trace(self, client: TestClient) -> None:
        provider = _FakeLLMProvider({IntentResult: _intent_result(), Plan: Plan(steps=[])})
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "Hi there."})

        assert response.status_code == 200
        trace = response.json()["execution_trace"]
        step_names = [s["step_name"] for s in trace["steps"]]
        assert "node:understand_intent" in step_names
        assert "node:synthesize" in step_names
        assert trace["total_duration_ms"] >= 0

    def test_session_id_echoed_back(self, client: TestClient) -> None:
        provider = _FakeLLMProvider({IntentResult: _intent_result(), Plan: Plan(steps=[])})
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "Hi.", "session_id": "sess-42"})

        assert response.status_code == 200
        assert response.json()["session_id"] == "sess-42"


class TestFileUpload:
    def test_query_with_text_file_ingests_and_returns_normalized_document(
        self, client: TestClient
    ) -> None:
        provider = _FakeLLMProvider(
            {IntentResult: _intent_result(intent=IntentType.SUMMARIZATION), Plan: Plan(steps=[])}
        )
        rag_service_mock = _override_agent_dependencies(client, provider)

        response = client.post(
            "/query",
            data={"query": "Summarize this file."},
            files=[("files", ("notes.txt", b"Quarterly revenue increased.", "text/plain"))],
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["normalized_documents"]) == 1
        assert data["normalized_documents"][0]["filename"] == "notes.txt"
        assert data["normalized_documents"][0]["content"] == "Quarterly revenue increased."
        # Ingested documents must be indexed into the (mocked) RAG service.
        rag_service_mock.index_documents.assert_called_once()


class TestValidation:
    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        provider = _FakeLLMProvider({IntentResult: _intent_result(), Plan: Plan(steps=[])})
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={})

        assert response.status_code == 422

    def test_blank_query_returns_400_invalid_input(self, client: TestClient) -> None:
        provider = _FakeLLMProvider({IntentResult: _intent_result(), Plan: Plan(steps=[])})
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "   "})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_INPUT"


class TestClarification:
    def test_ambiguous_request_returns_clarification_without_answer(self, client: TestClient) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(
                    is_ambiguous=True,
                    needs_clarification=True,
                    clarification_question="Which document do you mean?",
                ),
                Plan: Plan(steps=[]),
            }
        )
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "Summarize it."})

        assert response.status_code == 200
        data = response.json()
        assert data["clarification_needed"] is True
        assert data["clarification_prompt"] == "Which document do you mean?"
        assert data["answer"] is None


class TestIngestStillWorks:
    def test_ingest_endpoint_unaffected_by_new_agent_route(self, client: TestClient) -> None:
        response = client.post("/ingest", data={"text": "Plain text input."})
        assert response.status_code == 200
        assert response.json()["documents"][0]["content"] == "Plain text input."


class TestStatusAndErrorSurfacing:
    """Hardening pass: a tool/LLM failure the graph absorbs internally must
    still be visible in the response body (status/errors), not silently
    presented as an ordinary 200 with no indication anything went wrong."""

    def test_successful_run_reports_completed_status_and_no_errors(self, client: TestClient) -> None:
        provider = _FakeLLMProvider({IntentResult: _intent_result(), Plan: Plan(steps=[])})
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "Hi there."})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["errors"] == []

    def test_tool_failure_reports_failed_status_with_errors_but_still_200(
        self, client: TestClient
    ) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(intent=IntentType.QUESTION_ANSWERING),
                Plan: Plan(
                    steps=[
                        PlanStep(
                            step_id=0,
                            tool_name="rag_search",
                            purpose="Search for the answer.",
                            inputs={"query": "revenue"},
                            expected_result="Evidence found.",
                        )
                    ]
                ),
            },
            text_output="Sorry, I could not retrieve evidence for this request.",
        )
        rag_service_mock = _override_agent_dependencies(client, provider)
        rag_service_mock.retrieve.side_effect = RuntimeError("faiss exploded")
        # RAGSearchTool is bound to rag_service_mock via get_tool_registry's own
        # Depends(get_rag_service) chain, so overriding get_rag_service alone
        # (done above) is sufficient -- no separate tool_registry override needed.

        response = client.post("/query", data={"query": "What was revenue?"})

        assert response.status_code == 200  # the HTTP layer succeeded
        data = response.json()
        assert data["status"] == "failed"  # but the workflow reports its own failure
        assert data["errors"]  # non-empty, not silently dropped
        assert data["answer"] is not None  # still a best-effort answer, not nothing

    def test_clarification_reports_awaiting_clarification_status(self, client: TestClient) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _intent_result(
                    needs_clarification=True, clarification_question="Which one?"
                ),
                Plan: Plan(steps=[]),
            }
        )
        _ = _override_agent_dependencies(client, provider)

        response = client.post("/query", data={"query": "Summarize it."})

        assert response.status_code == 200
        assert response.json()["status"] == "awaiting_clarification"


class TestMissingGeminiConfiguration:
    """Hardening pass: a real dependency-construction failure (no
    GEMINI_API_KEY configured) must surface as a clean JSON error via the
    existing exception handlers, never an unhandled 500 crash -- exercises
    get_llm_provider()'s REAL path (not overridden), unlike every other
    test in this file."""

    def test_missing_gemini_api_key_returns_clean_configuration_error(
        self, client: TestClient
    ) -> None:
        client.app.dependency_overrides[get_rag_service] = lambda: MagicMock()
        # get_llm_provider is deliberately NOT overridden here.

        response = client.post("/query", data={"query": "Hello"})

        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "CONFIGURATION_ERROR"


class TestEmbeddingModelCaching:
    """Hardening pass: the embedding model must not be reloaded from disk on
    every request -- get_embedding_service is a process-wide cache."""

    def test_get_embedding_service_returns_the_same_cached_instance(self) -> None:
        first = get_embedding_service()
        second = get_embedding_service()
        assert first is second

    def test_get_rag_service_is_constructed_with_the_cached_embedding_service(self) -> None:
        from omniflow.api.routes.agent import get_rag_service

        shared_embedding_service = get_embedding_service()
        rag_service = get_rag_service(embedding_service=shared_embedding_service)
        # White-box check of our own wiring: confirms get_rag_service actually
        # threads the cached instance through to RAGService rather than
        # silently constructing its own (which would defeat the cache).
        assert rag_service._embedding_service is shared_embedding_service

    def test_embedding_service_is_a_real_embedding_service_instance(self) -> None:
        assert isinstance(get_embedding_service(), EmbeddingService)
