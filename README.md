# OmniFlow: Agentic Multimodal AI Assistant

OmniFlow is an agentic multimodal assistant engineered for intelligent ingestion, autonomous planning, deterministic tool execution, conditional RAG, and multi-input synthesis.

> **Current Status: Phase 1 (Foundation)**  
> This repository currently contains the Phase 1 architectural foundation. Multimodal processing (OCR, PDF extraction, speech-to-text, YouTube transcript scraping), vector RAG, LangGraph orchestration, LLM reasoning, and UI components are scheduled for implementation in subsequent phases.

---

## Architectural Overview

The target end-to-end architecture follows a strictly deterministic-first, LLM-reasoning pipeline:

```
User Request / Multi-modal Uploads
              │
              ▼
   FastAPI Gateway (Modular Routes)
              │
              ▼
   Multimodal Ingestion (Modality Processors)
              │
              ▼
   Normalized Documents (Unified Schema)
              │
              ▼
   Unified Context & Intent Classifier
              │
              ▼
   LangGraph Orchestrator (Stateful Graph)
              │
      ┌───────┴───────┐
      ▼               ▼
Deterministic    Conditional
Python Tools     FAISS Vector RAG
      │               │
      └───────┬───────┘
              ▼
   Gemini LLM Synthesizer
              │
              ▼
   Structured OmniFlow Response
```

### Architectural Principles
- **Modality-agnostic Normalization**: Raw inputs (text, images, PDFs, audio) are ingested into normalized domain models (`NormalizedDocument`) before reaching the agent.
- **Deterministic Separation**: File parsing, OCR, audio transcription, URL detection, and mathematical/tabular operations use deterministic Python routines—the LLM is reserved for semantic planning and final synthesis.
- **Safe Execution Traces**: Tracing captures non-sensitive operational metadata (operation name, status, duration) without logging credentials, full prompts, or raw binary payloads.
- **Free-Tier Friendly**: Designed for local and zero-cost cloud deployment (FAISS local vector storage, standard Python runtimes, no mandatory paid SaaS).

---

## Phase 1 Components

The following foundation has been established in Phase 1:

- **Application Entry Point**: Clean FastAPI application factory in `omniflow/main.py` with lifespan event management.
- **API Endpoints**: Modular routing featuring `GET /health` and `GET /` discovery endpoints.
- **Configuration**: Centralized `pydantic-settings` loader in `omniflow/config.py` reading from environment variables with `.env.example`.
- **Domain Models**: Robust Pydantic schemas in `omniflow/models/` for `NormalizedDocument`, `UserRequest`, `OmniFlowResponse`, `ExecutionTrace`, and `AgentState`.
- **Architectural Contracts**: Minimal, unbloated abstract base interfaces for `BaseProcessor`, `BaseLLMProvider`, `BaseTool`, and `BaseVectorStore`.
- **Exception Hierarchy**: Domain error classes (`InvalidInputError`, `UnsupportedFileError`, `ProcessingFailureError`, `ExternalProviderError`, `OrchestrationError`, `ConfigurationError`) in `omniflow/exceptions.py`.
- **API Error Handling**: Global exception handlers in `omniflow/api/error_handlers.py` converting domain exceptions to uniform JSON responses without exposing internal stack traces.
- **Centralized Logging**: Production-ready logging in `omniflow/logging.py` featuring execution time tracking and automated secret sanitization.
- **Automated Tests**: Unit and integration test suite in `tests/` covering health checks, configuration, model validation, and exception handling.

---

## Getting Started

### 1. Prerequisites
- Python 3.11 or 3.12
- pip package manager

### 2. Create and Activate a Virtual Environment

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

Install the minimal Phase 1 requirements:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file from the provided template:
```bash
cp .env.example .env
```

Review `.env` and adjust settings as needed:
```env
APP_ENV=development
LOG_LEVEL=INFO
HOST=127.0.0.1
PORT=8000
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.5-flash
MAX_UPLOAD_SIZE_MB=25
```

> [!NOTE]
> In Phase 1, `GEMINI_API_KEY` is not required for application startup or running tests. It will be required once LLM providers are introduced in Phase 2.

### 5. Run the FastAPI Application

Start the local development server:
```bash
uvicorn omniflow.main:app --reload --host 127.0.0.1 --port 8000
```

Once running, access:
- **Root endpoint**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Health check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive OpenAPI docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 6. Run the Test Suite

Execute the automated tests using `pytest`:
```bash
pytest -v
```

---

## Roadmap & Subsequent Phases

- **Phase 2: Multimodal Ingestion Processors**
  - Native text PDF extraction with Tesseract OCR fallback
  - Image preprocessing and OCR
  - Audio transcription (STT)
  - Content-type routing and document normalization
- **Phase 3: Deterministic Tools & Retrieval**
  - YouTube URL detection and transcript extraction
  - Local FAISS vector indexing and conditional RAG
- **Phase 4: Agent Orchestration & Planning**
  - LangGraph workflow construction
  - Intent classification and ambiguity clarification check
  - Minimum-tool planner and execution engine
  - Provider integration with Google Gemini API
- **Phase 5: Synthesis, User Interface & Deployment**
  - Cross-input synthesis and execution trace reporting
  - Lightweight UI for text, image, PDF, and audio uploads
  - Docker containerization and cloud deployment
