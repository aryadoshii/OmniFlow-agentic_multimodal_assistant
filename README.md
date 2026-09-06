# OmniFlow: Agentic Multimodal AI Assistant

OmniFlow is an agentic multimodal assistant engineered for intelligent ingestion, autonomous planning, deterministic tool execution, conditional RAG, and multi-input synthesis.

> **Current Status: Phase 3 Implemented (Deterministic Tool + RAG Layer)**  
> This repository contains the Phase 1 architectural foundation, Phase 2 multimodal ingestion layer, and Phase 3 deterministic Tool Framework + RAG pipeline (chunking, local embeddings, FAISS retrieval, YouTube transcripts). LangGraph orchestration, LLM reasoning/synthesis, and UI components are scheduled for subsequent phases.

---

## Status Summary

### Implemented
- **Phase 1: Architecture & Foundation**: Modular FastAPI structure, domain models (`NormalizedDocument`, `AgentState`), centralized configuration, exception hierarchy, error handlers, and logging.
- **Phase 2: Multimodal Ingestion Layer**: Deterministic processors converting plain text, PDFs (native text with per-page OCR fallback), images (JPG/PNG via Tesseract OCR), and audio (WAV/MP3/M4A via faster-whisper) into unified `NormalizedDocument` representations.
- **Phase 3: Deterministic Tool + RAG Layer**: A structured-I/O tool abstraction and registry (`BaseTool`, `ToolRegistry`); a YouTube transcript retrieval tool (no API key required); a local, lazy-loaded embedding service (`sentence-transformers/all-MiniLM-L6-v2`); deterministic document chunking with source/metadata preservation; an in-memory FAISS vector store; a `RAGService` that turns ingested documents into ranked, source-attributed evidence (or an explicit "insufficient evidence" result — never a hallucinated answer); and a `RAGSearchTool` exposing that pipeline through the same tool registry as YouTube retrieval. See [Phase 3: Tool + RAG Architecture](#phase-3-tool--rag-architecture) below.

### Not Yet Implemented (Scheduled for Phase 4+)
- LangGraph orchestration, state machine & planner (deciding *when* to call which tool)
- Ambiguity clarification flow
- Gemini LLM reasoning & synthesis (Phase 3's RAG output is evidence, not an answer)
- Frontend web UI
- Docker containerization & cloud deployment

---

## Ingestion Architecture

```
User Text / Uploaded Files (PDF, Image, Audio)
                     │
                     ▼
             POST /ingest API
                     │
                     ▼
          FileValidator & Limits Check
          (Size, MIME, Extension, Non-empty)
                     │
                     ▼
              IngestionService
         (Deterministic Dispatcher)
                     │
     ┌───────────────┼───────────────┬───────────────┐
     ▼               ▼               ▼               ▼
TextProcessor   PDFProcessor    ImageProcessor  AudioProcessor
 (Plain text)    (PyMuPDF)       (PIL + OCR)    (faster-whisper)
                     │
                     ▼
          Native Text >= Threshold?
             ├── Yes ──► Native text
             └── No  ──► Per-page Tesseract OCR fallback
                     │
                     ▼
        NormalizedDocument (Unified Model)
  (Content, SourceType, ExtractionMethod, Metadata, Warnings)
```

### Key Ingestion Strategies
- **Deterministic Processor Selection**: Files are routed to processors based on a practical combination of file extension, declared MIME type, and processor format decoders.
- **PDF Native/OCR Hybrid Fallback**: Every page is analyzed natively with PyMuPDF. If a page has meaningful text (>= threshold), native text is used. If a page is scanned or empty, PyMuPDF renders the page to an image and runs Tesseract OCR. Mixed PDFs are accurately labeled `ExtractionMethod.MIXED`.
- **Lazy & Optional Speech-to-Text**: `faster-whisper` is loaded purely on demand when an audio file is processed. The app starts and handles text/PDF/images with zero Whisper overhead. Model size (`tiny` by default) and compute type (`int8`) are fully configurable for CPU/free-tier constraints.
- **Safe Ephemeral Storage**: Uploads are processed in system temporary files outside the repository tree, guaranteeing cleanup upon completion or failure.

---

## Phase 3: Tool + RAG Architecture

```
ToolRegistry
    ├── youtube_transcript  ──► YouTubeTranscriptTool ──► youtube-transcript-api (no API key)
    └── rag_search          ──► RAGSearchTool ──► RAGService
                                                       │
                              NormalizedDocument ──► DocumentChunker ──► DocumentChunk(s)
                                                       │
                                                  EmbeddingService (local, lazy-loaded)
                                                       │
                                              FAISSVectorStore (in-memory, IndexFlatIP)
                                                       │
                                        RAGResult (ranked evidence, or explicit "no evidence")
```

**Design principles:**
- **Structured I/O everywhere**: every tool declares a Pydantic `input_model` and returns a Pydantic output — never a raw dict. `ToolRegistry.execute()` validates input and enforces structured output before returning `(result, ToolExecutionTrace)`.
- **Evidence, not answers**: `RAGService.retrieve()` and `RAGSearchTool` never call an LLM and never generate an answer. If nothing indexed clears the similarity threshold, the result explicitly reports `has_evidence=False` / `status="no_evidence"` with a human-readable reason — a deliberate refusal to hand a future LLM synthesis step irrelevant context, rather than a silent empty list.
- **Local, keyless, and lazy**: embeddings run locally via `sentence-transformers/all-MiniLM-L6-v2` (~80MB, CPU-friendly, no API key); the model is not imported or loaded until the first embedding call. YouTube transcripts are fetched via the free `youtube-transcript-api` library — no paid API, no official YouTube Data API key.
- **Full source attribution**: every retrieved chunk carries its originating `document_id`, `filename`, `source_type`, `extraction_method`, chunk position, and any page/segment metadata the ingestion layer already produced (e.g. a PDF's per-page breakdown) — traceable all the way back to the original upload.
- **No orchestration logic yet, by design**: `ToolRegistry` and `RAGService` are a deterministic *execution substrate* — they don't decide *when* to call `rag_search` vs. `youtube_transcript`, or chain tools together. That decision logic is explicitly Phase 4's job (LangGraph); Phase 3 only guarantees that once a tool is called, it behaves safely, predictably, and with clear, typed error reporting.

**Configuration** (`.env` / `omniflow/config.py`): `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_SIMILARITY_THRESHOLD`, `YOUTUBE_TRANSCRIPT_MAX_CHARS` — see `.env.example` for defaults.

**Not yet wired to any HTTP route.** `ToolRegistry`, `RAGService`, and `RAGSearchTool` are fully implemented, tested, and usable programmatically, but no FastAPI endpoint currently exposes them (only `POST /ingest` exists as a live route) — that wiring is expected to arrive alongside Phase 4's orchestration layer.

---

## Supported Input Formats

| Modality | Supported Formats | Engine / Library | Extraction Method |
|---|---|---|---|
| **Text** | `.txt` | Native Python | `direct_input` |
| **PDF** | `.pdf` | PyMuPDF (`pymupdf`) + OCR fallback | `native_text`, `ocr`, or `mixed` |
| **Image** | `.jpg`, `.jpeg`, `.png` | Pillow + Tesseract (`pytesseract`) | `ocr` |
| **Audio** | `.wav`, `.mp3`, `.m4a` | faster-whisper | `speech_to_text` |

---

## Local & System Dependencies

In addition to Python packages, the following local tools are supported for full local capability:
- **Tesseract OCR**: Required for image and scanned PDF OCR (`brew install tesseract` on macOS or `apt-get install tesseract-ocr` on Linux). If absent, PDF native text extraction continues to operate, while OCR requests raise a clear `OCRProcessingError`.
- **FFmpeg**: Utilized by audio decoders (`brew install ffmpeg` on macOS or `apt-get install ffmpeg` on Linux).

---

## Getting Started

### 1. Prerequisites
- Python 3.11 or 3.12
- pip package manager
- (Optional for OCR/Audio): Tesseract OCR and FFmpeg

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your `.env` file:
```bash
cp .env.example .env
```

Key settings (see `.env.example` for the complete, commented list):
```env
APP_ENV=development
LOG_LEVEL=INFO
HOST=127.0.0.1
PORT=8000
MAX_UPLOAD_SIZE_MB=25

# OCR Configuration
OCR_LANGUAGE=eng
TESSERACT_CMD=
PDF_NATIVE_TEXT_CHAR_THRESHOLD=30

# Speech-to-Text (faster-whisper) Configuration
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Phase 3: Embedding Configuration (local, no API key)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

# Phase 3: RAG Chunking / Retrieval Configuration
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=4
RAG_SIMILARITY_THRESHOLD=0.2

# Phase 3: YouTube Transcript Tool Configuration
YOUTUBE_TRANSCRIPT_MAX_CHARS=200000
```

### 5. Run the Application

```bash
uvicorn omniflow.main:app --reload --host 127.0.0.1 --port 8000
```

- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Ingestion Endpoint**: `POST http://127.0.0.1:8000/ingest`
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 6. Ingestion API Example

Using `curl`:
```bash
# Ingest plain text
curl -X POST http://127.0.0.1:8000/ingest \
  -F "text=Summarize this quarterly update."

# Ingest multi-modal files
curl -X POST http://127.0.0.1:8000/ingest \
  -F "text=Review these documents." \
  -F "files=@financial_report.pdf" \
  -F "files=@receipt.png"
```

### 7. Phase 3 Tool + RAG Usage Example (programmatic, no HTTP route yet)

```python
from omniflow.rag.service import RAGService
from omniflow.tools.registry import ToolRegistry
from omniflow.tools.rag_search import RAGSearchTool
from omniflow.tools.youtube import YouTubeTranscriptTool
from omniflow.models.document import NormalizedDocument, SourceType, ExtractionMethod

rag_service = RAGService()
registry = ToolRegistry()
registry.register(YouTubeTranscriptTool())
registry.register(RAGSearchTool(rag_service))

doc = NormalizedDocument(
    filename="notes.txt",
    source_type=SourceType.TEXT,
    mime_type="text/plain",
    content="OmniFlow's RAG pipeline chunks, embeds, and indexes documents locally.",
    extraction_method=ExtractionMethod.DIRECT_INPUT,
)
rag_service.index_documents([doc])

output, trace = registry.execute("rag_search", query="How does the RAG pipeline work?")
print(output.status)     # "evidence_found" or "no_evidence" -- never a generated answer
print(output.evidence)   # ranked, source-attributed RetrievedChunk list
```

### 8. Run the Test Suite

```bash
pytest -v
```

The default suite (214 tests as of Phase 3) is fully mocked at the true external boundary (the embedding model, the YouTube API) and runs in well under a second with no network access or model downloads. Two additional tests exercise the **real** embedding model end-to-end and are skipped by default; run them explicitly once the model has been downloaded (first run only) via:

```bash
OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS=1 pytest -k RealModel -v
```

---

## Phase 3 Known Limitations

- **Not yet wired to any HTTP route or orchestrator.** `RAGService` and both tools are fully functional and tested as standalone components, but nothing currently calls them from `/ingest` or any other endpoint — that integration is Phase 4's job.
- **`faiss-cpu` and `sentence-transformers` (torch) can conflict if `faiss` is imported first in the same process**, due to a third-party OpenMP runtime conflict (observed on macOS; not an OmniFlow bug). Import `torch` (or `sentence_transformers`) before `faiss` anywhere both are used together — this repository's own code already does so correctly via lazy, on-demand imports inside each service.
- **No metadata-based filtering in retrieval yet** (e.g. restricting a search to one document or source type) — every query searches the full index.
- **No persistence.** The FAISS index and chunk metadata are in-memory only and are lost on process restart, by design for this phase.
- **The default similarity threshold (0.2) is a conservative, principled default, not an empirically tuned one** — no labeled relevance dataset or downstream answer-quality signal exists yet to validate it against.
- YouTube transcript retrieval has been verified against a mocked provider boundary in all automated tests; this README does not claim live network retrieval from youtube.com has been exercised in this environment.

---

## Deployment (Docker + Render)

A single-service deployment: one Docker container runs the FastAPI backend
and also serves the built React frontend from the same process/origin
(see `omniflow/main.py` — `GET /` serves the built SPA only when
`APP_ENV=production` **and** the frontend has actually been built into
`frontend/dist`; local/dev/test runs are unaffected and keep the existing
JSON discovery response at `/`).

**Live demo:** `https://<your-render-service-name>.onrender.com` — replace
this placeholder with your actual Render URL once deployed. (Render's free
tier spins the service down after inactivity; the first request after a
period of idleness can take 30–60+ seconds to respond while it wakes up.)

### Build and run with Docker locally

```bash
docker build -t omniflow .
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=your_real_key_here \
  omniflow
```

Then visit:
- `http://localhost:8000/` — the built frontend (single-page app)
- `http://localhost:8000/health` — health check
- `http://localhost:8000/docs` — OpenAPI docs

The image builds the frontend in a Node stage and copies only the built
static output into the final Python image — Node itself is not present in
the runtime image. Without `GEMINI_API_KEY` set, the container still starts
and serves the UI/`/health`/`/ingest` normally; `/query` will return a
clean `CONFIGURATION_ERROR` JSON response until a real key is supplied
(this is the existing, tested behavior — not new for deployment).

### Deploying to Render

**Option A — Blueprint (recommended):** In the Render dashboard, choose
**New +** → **Blueprint**, point it at this repository. Render reads
`render.yaml` at the repo root and provisions a single Docker-based Web
Service (free plan, health check at `/health`). After it's created, open
the service's **Environment** tab and set `GEMINI_API_KEY` to your real
key (deliberately left unset in `render.yaml` — never commit it).

**Option B — Manual dashboard setup:** New + → Web Service → connect this
repo → Runtime: **Docker** → leave the Dockerfile path as the repo root
`Dockerfile` → plan: **Free** → Health Check Path: `/health` → add the
environment variables listed below → Create Web Service.

Render automatically supplies `$PORT`; the Dockerfile's `CMD` reads it at
container start (`uvicorn ... --port ${PORT:-8000}`) — do not set `PORT`
yourself.

### Required/optional environment variables (Render)

| Variable | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | Set only in Render's dashboard (or via `render.yaml` with `sync: false`, filled in manually) — never committed. Without it, `/query` returns a typed `CONFIGURATION_ERROR`, not a crash. |
| `APP_ENV` | Recommended | Set to `production` — this is what makes `GET /` serve the built frontend (see above). `render.yaml` sets this for you. |
| `LOG_LEVEL` | No | Defaults to `INFO`. |
| `LLM_MODEL` | No | Defaults to `gemini-2.5-flash`. |
| `GEMINI_TIMEOUT_SECONDS` | No | Defaults to `30`. |
| `MAX_UPLOAD_SIZE_MB` | No | Defaults to `25`. |
| `MAX_AGENT_STEPS` / `MAX_TOOL_CALLS` / `MAX_RETRIES` | No | Bound the agent loop; existing defaults (`6`/`6`/`2`) preserved — see "Resource / memory limitations" below before lowering further. |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_DEVICE` / `EMBEDDING_BATCH_SIZE` | No | Existing RAG/embedding defaults preserved unchanged; see limitations below. |
| `WHISPER_MODEL_SIZE` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | No | Existing defaults (`tiny`/`cpu`/`int8`) preserved — already the lightest available Whisper configuration. |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` / `RAG_TOP_K` / `RAG_SIMILARITY_THRESHOLD` | No | Existing RAG defaults preserved unchanged. |
| `PORT` | **Do not set** | Supplied automatically by Render at container start. |
| `HOST` | No | The Dockerfile already sets this to `0.0.0.0` (required to accept Render's traffic); no action needed. |

See `.env.example` for the full list with descriptions (used for local
development; Render reads its own dashboard/`render.yaml` variables, not
this file).

### Health check

`GET /health` returns `{"status": "healthy", "app_name": "OmniFlow", "version": "...", "environment": "..."}` with HTTP 200 and requires no
authentication or request body — configured as Render's health check path
in `render.yaml` (and should be entered the same way if configuring
manually).

### Resource / memory limitations (read before relying on RAG in production)

- **Lazy loading is preserved everywhere**: the sentence-transformers
  embedding model and faster-whisper are never loaded at container
  startup, and RAG indexing only happens on a request that actually
  invokes the `rag_search` tool (see Phase 5's lazy-indexing fix) — a
  direct-context request (e.g. "summarize this short PDF") never touches
  either, regardless of file uploads.
- **The first request that genuinely triggers RAG will load
  sentence-transformers/PyTorch into memory** (observed ~600 MB–1 GB
  resident memory during development). Render's free tier is memory
  constrained; if the service is killed or restarts unexpectedly on such a
  request, this is the most likely cause. This is a capacity/hosting-tier
  limitation, not an application defect — see Phase 5/6's reports for the
  full investigation. Upgrading the Render plan, or a future model swap
  (explicitly out of scope for this phase), would address it; no code
  change was made here to work around it.
- Audio transcription (faster-whisper, `tiny` model) was confirmed to stay
  well under that ceiling in the same environment.
