# OmniFlow: Agentic Multimodal AI Assistant

OmniFlow is an agentic multimodal assistant: it ingests text, PDFs, images, and audio
(individually or together), understands the user's intent and constraints, asks a
clarifying question when the request is genuinely ambiguous, autonomously plans and
executes a minimum-viable sequence of deterministic tools (semantic search over the
uploaded documents, YouTube transcript retrieval), and synthesizes a single grounded,
text-only answer — with a safe, human-readable trace of every step it took.

> **Current status: Phases 1–7 implemented.** Multimodal ingestion, the deterministic
> tool + RAG layer, the LangGraph agent (intent understanding, clarification,
> planning, bounded tool execution/replanning, synthesis, structural validation), the
> React frontend, and a single-service Docker/Render deployment path are all built and
> tested. **Phase 8 (live deployment + end-to-end verification against the five
> official assignment scenarios) is the remaining step** — see
> [Deployment](#deployment-docker--render) and [Known limitations](#known-limitations)
> below.

---

## Status summary

### Implemented

- **Foundation**: modular FastAPI structure, domain models (`NormalizedDocument`,
  `AgentState`, `OmniFlowResponse`), centralized configuration, exception hierarchy,
  uniform error handlers, and logging.
- **Multimodal ingestion**: deterministic processors converting plain text, PDFs
  (native text with per-page OCR fallback), images (JPG/PNG via Tesseract), and audio
  (WAV/MP3/M4A via `faster-whisper`) into a unified `NormalizedDocument` representation
  — including OCR confidence scores and audio duration/language metadata.
- **Deterministic tool + RAG layer**: a structured-I/O tool abstraction and registry
  (`BaseTool`, `ToolRegistry`); a YouTube transcript retrieval tool (no API key
  required, all four URL formats supported); a local, lazy-loaded embedding service
  (`sentence-transformers/all-MiniLM-L6-v2`); from-scratch recursive document chunking;
  an in-memory FAISS vector store; and a `RAGService` that returns ranked,
  source-attributed evidence or an explicit "no evidence" result — never a
  hallucinated answer.
- **LangGraph agent**: a bounded plan → execute → observe → replan loop with three
  independent termination guarantees (`MAX_AGENT_STEPS`, `MAX_TOOL_CALLS`,
  `MAX_RETRIES`), real intent classification and ambiguity detection via Gemini
  structured output, a planner that never invents a tool name and never repeats a
  failing tool call forever, single-call synthesis grounded in aggregated document
  context and tool results, and one bounded structural-correction pass if the output
  violates a user-stated constraint (e.g. an exact bullet count).
- **Frontend**: a React/Vite/TypeScript chat-style UI — multi-file upload, a
  clarification-aware query box (composes a self-contained follow-up query client-side
  since the backend is stateless per request), a human-readable execution trace panel,
  and error/clarification states.
- **Deployment wiring**: a multi-stage Dockerfile (Node build stage → Python runtime),
  a `render.yaml` Blueprint, and production-only single-origin frontend serving from
  the same FastAPI process (see [Deployment](#deployment-docker--render)).

### Not yet done

- **Live public deployment.** The Docker/Render configuration exists and has been
  validated locally (see below), but the service has not yet been deployed to a live
  Render URL, and the five official assignment scenarios have not yet been run
  end-to-end against a real Gemini key in that environment.
- **A structurally-enforced summarization contract.** The assignment requires every
  summarization/transcription-summary response to include a 1-line summary, 3
  bullets, and a 5-sentence summary, *simultaneously, by default* — not only when the
  user explicitly asks for that shape. Today, `validate_structure()` only checks
  constraints the user stated verbatim; the triple-format shape is currently only
  encouraged by the synthesis system prompt, not deterministically enforced. Treat
  this as an open correctness gap against Test Case 1, not a documented non-goal.
- **Deterministic YouTube URL detection.** `NormalizedDocument.detected_urls` exists
  in the schema but is never populated by any processor; a URL inside a document is
  currently only found if the intent-understanding LLM notices it in the (bounded)
  content preview. Works, but has no deterministic regex fallback.
- **Multi-turn state.** `session_id` is accepted and echoed back, but the backend
  keeps no server-side session store — every `/query` call builds a fresh
  `AgentState`. The frontend works around this for clarification follow-ups by
  composing a single self-contained query client-side (original request + question +
  answer) and resubmitting it with the same files.

---

## Ingestion architecture

```
User Text / Uploaded Files (PDF, Image, Audio)
                     │
                     ▼
        POST /query or POST /ingest
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
  (Content, SourceType, ExtractionMethod, Confidence, Metadata, Warnings)
```

- **Deterministic processor selection** by file extension + declared MIME type.
- **PDF native/OCR hybrid fallback**: every page is checked natively with PyMuPDF
  first; a page below the character threshold is rendered to an image and OCR'd with
  Tesseract instead. Mixed PDFs are labeled `ExtractionMethod.MIXED`. A single page's
  OCR failure never destroys extraction for the rest of the document.
- **Lazy, optional STT**: `faster-whisper` loads only on the first audio request, not
  at startup. Model size (`tiny` by default) and compute type (`int8`) are
  configurable for CPU/free-tier constraints.
- **Safe ephemeral storage**: uploads are written to sanitized, path-traversal-safe
  temporary files and always removed after processing, success or failure.

---

## Agent architecture (LangGraph)

```
START → prepare_context → understand_intent → check_clarity
                                                     │
                               ┌─────────────────────┴─────────────────────┐
                               ▼                                           ▼
                         clarification                                   plan
                               │                                           │
                              END                                    execute_tool
                                                                            │
                                                                     observe_result
                                                                            │
                                                                       route_next
                                                                            │
                                                  ┌─────────────────────────┴─────────────────────────┐
                                                  ▼ (more work)                                        ▼ (complete)
                                                plan                                              synthesize
                                                                                                        │
                                                                                                validate_output
                                                                                                        │
                                               ┌────────────────────────────────────────────────────────┴──┐
                                               ▼ (structural violation, budget allows)                     ▼ (valid, or budget exhausted)
                                            synthesize                                                    END
```

- **`understand_intent`**: a single Gemini structured-output call classifies intent,
  extracts constraints verbatim, resolves cross-input references (deterministically
  for unambiguous cases like "this PDF" with exactly one PDF present; via the LLM for
  genuinely ambiguous or cross-document references), and decides whether the request
  needs clarification.
- **`plan`**: produces the minimum-viable next step(s), constrained to tools that
  actually exist in the `ToolRegistry` — a hallucinated tool name is treated as a
  planning failure, never silently executed. Re-invoked after every tool call
  (replanning), aware of everything executed so far, so a "no evidence found" result
  is accepted as a final answer rather than retried forever.
- **`execute_tool`**: dispatches to the `ToolRegistry`. Three independent,
  `Settings`-driven bounds (`MAX_AGENT_STEPS`, `MAX_TOOL_CALLS`, `MAX_RETRIES`)
  guarantee the loop always terminates regardless of what the planner decides.
- **`synthesize`**: one plain-text Gemini call grounded in the aggregated document
  context and every tool result so far; explicitly instructed to say "no evidence
  found" rather than fabricate when retrieval comes up empty.
- **`validate_output`**: deterministic, regex-based structural checks (bullet counts,
  sentence counts, one-line output, named required sections) against the constraints
  the user actually stated — never a semantic-correctness check. One bounded
  correction pass is allowed if a violation is found.

Tools currently registered: `youtube_transcript` (no API key; supports watch,
`youtu.be`, `/shorts/`, and `/embed/` URL formats) and `rag_search` (semantic search
over the current request's own uploaded documents — indexing is lazy, so a
direct-context request never loads the embedding backend just because files were
attached).

---

## Supported input formats

| Modality | Supported Formats | Engine / Library | Extraction Method |
|---|---|---|---|
| **Text** | `.txt`, plain query text | Native Python | `direct_input` |
| **PDF** | `.pdf` | PyMuPDF (`pymupdf`) + OCR fallback | `native_text`, `ocr`, or `mixed` |
| **Image** | `.jpg`, `.jpeg`, `.png` | Pillow + Tesseract (`pytesseract`) | `ocr` |
| **Audio** | `.wav`, `.mp3`, `.m4a` | faster-whisper | `speech_to_text` |

---

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check (used as Render's health check path). |
| `POST /ingest` | Lower-level: normalizes text/files into `NormalizedDocument`s only. No agent, no LLM call. |
| `POST /query` | The real entry point: runs the full agent workflow (ingest → intent → clarify-or-plan → bounded execute/replan → synthesize → validate) and returns an `OmniFlowResponse`. |
| `GET /docs` | Interactive OpenAPI documentation. |
| `GET /` | In production with a built frontend present, serves the React SPA. Otherwise, a JSON discovery response. |

---

## Local & system dependencies

- **Tesseract OCR** — required for image and scanned-PDF OCR
  (`apt-get install tesseract-ocr` on Linux, `brew install tesseract` on macOS). If
  absent, native PDF text extraction still works; an OCR request raises a clear
  `OCRProcessingError`.
- **FFmpeg** — used by `faster-whisper`'s audio decoding
  (`apt-get install ffmpeg` / `brew install ffmpeg`).
- **libgomp1** (Linux/Docker only) — the OpenMP runtime `faiss-cpu`/`torch` need on
  Debian-slim images; already installed in the provided Dockerfile.

---

## Getting started (local development)

### 1. Prerequisites

- Python 3.11 or 3.12
- Node.js 20+ (only if you want to run the frontend)
- Tesseract OCR and FFmpeg (optional locally; required for OCR/audio)

### 2. Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GEMINI_API_KEY to run the real /query agent workflow
uvicorn omniflow.main:app --reload --host 127.0.0.1 --port 8000
```

- Health check: http://127.0.0.1:8000/health
- Interactive docs: http://127.0.0.1:8000/docs
- Agent endpoint: `POST http://127.0.0.1:8000/query` (form fields: `query`, optional
  `session_id`, optional `files`)

### 3. Frontend setup (optional, for local UI development)

```bash
cd frontend
npm install
npm run dev
```

Vite's dev server proxies `/api/*` to the local backend (see `vite.config.ts`); no
`VITE_API_BASE_URL` is needed for local development.

### 4. Example requests

```bash
# Ingestion only (no agent, no LLM call)
curl -X POST http://127.0.0.1:8000/ingest \
  -F "text=Summarize this quarterly update."

# Full agent workflow
curl -X POST http://127.0.0.1:8000/query \
  -F "query=What are the action items in this document?" \
  -F "files=@meeting_notes.pdf"
```

### 5. Run the test suite

```bash
pytest -q
```

453 tests pass, 2 are skipped by default (they exercise the **real** embedding model
end-to-end rather than a mock, and require the model to have been downloaded once).
Run them explicitly with:

```bash
OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS=1 pytest -k RealModel -v
```

The default suite mocks every true external boundary (Gemini, the embedding model,
the YouTube API, Whisper) and needs no network access or model downloads.

### 6. Lint

```bash
ruff check .          # backend — see note below
cd frontend && npm run lint   # frontend (oxlint) — currently clean
```

**Note on backend lint:** this repository has no `pyproject.toml`/`ruff.toml`, so
`ruff check .` runs against ruff's full default rule set. As of this writing that
surfaces ~85 findings (mostly import-sort ordering and a handful of `SIM117`
nested-`with` suggestions in the test files — style preferences, not correctness
bugs, and 58 of the 85 are auto-fixable with `ruff check . --fix`). If you want a
lint gate that reflects the ruleset this project was actually written against, add a
`pyproject.toml` pinning the intended `select`/`ignore` set before treating
`ruff check .` as pass/fail.

---

## Deployment (Docker + Render)

A single-service deployment: one Docker container builds the React frontend in a
Node stage, then serves both the built static assets and the FastAPI backend from
the same Python runtime image and the same origin — so there is no cross-origin
API traffic and no CORS configuration is needed. `GET /` serves the built SPA only
when `APP_ENV=production` **and** a build exists at `frontend/dist`; local/dev/test
runs are unaffected and keep the existing JSON discovery response at `/`.

**Live demo:** _not yet deployed — replace this line with your Render URL once
Phase 8 is complete._ (Render's free tier spins the service down after inactivity;
the first request after a period of idleness can take 30–60+ seconds while it wakes
up.)

### Build and run with Docker locally

```bash
docker build -t omniflow .
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=your_real_key_here \
  omniflow
```

Then visit `http://localhost:8000/` (frontend), `/health`, and `/docs`. Without
`GEMINI_API_KEY` set, the container still starts and serves the UI/`/health`/`/ingest`
normally — `/query` returns a typed `CONFIGURATION_ERROR` JSON response instead of
crashing.

The image builds the frontend in a `node:20-slim` stage and copies only the built
static output into the `python:3.12-slim` runtime — Node is never present in the
final image. The Python dependency install explicitly points at PyTorch's CPU-only
wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`); without this,
`sentence-transformers`' default resolution of its `torch` dependency pulls in the
CUDA build, inflating the image from roughly 3.2GB to over 10GB for GPU libraries
that are never used (both `EMBEDDING_DEVICE` and `WHISPER_DEVICE` default to `cpu`).

### Deploying to Render

**Option A — Blueprint (recommended).** Push this repository to GitHub with
`render.yaml`, `Dockerfile`, and `.dockerignore` committed. In the Render dashboard:
**New +** → **Blueprint** → select the repo. Render reads `render.yaml` and
provisions a Docker-based free Web Service with a health check at `/health`. Open the
new service's **Environment** tab and set `GEMINI_API_KEY` to your real key
(deliberately left blank in `render.yaml` via `sync: false` — never committed).
Deploy.

**Option B — Manual dashboard setup.** New + → Web Service → connect the repo →
Runtime: **Docker** → Health Check Path: `/health` → Plan: **Free** → add
`GEMINI_API_KEY` under Environment → Create Web Service.

Render supplies `$PORT` automatically; the Dockerfile's `CMD` reads it
(`uvicorn ... --port ${PORT:-8000}`) — do not set `PORT` yourself.

### Environment variables (Render)

| Variable | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | Set only in Render's dashboard — never committed. Without it, `/query` returns a typed `CONFIGURATION_ERROR`, not a crash. |
| `APP_ENV` | Set by `render.yaml` | `production` — this is what makes `GET /` serve the built frontend. |
| `PORT` | **Do not set** | Supplied automatically by Render at container start. |
| `HOST` | No | The Dockerfile already sets this to `0.0.0.0`; no action needed. |
| `LOG_LEVEL`, `LLM_MODEL`, `GEMINI_TIMEOUT_SECONDS`, `MAX_UPLOAD_SIZE_MB`, `MAX_AGENT_STEPS`, `MAX_TOOL_CALLS`, `MAX_RETRIES`, `EMBEDDING_*`, `WHISPER_*`, `RAG_*`, `YOUTUBE_TRANSCRIPT_MAX_CHARS` | No | Existing defaults preserved; see `.env.example` for the full list. |

### Health check

`GET /health` returns `{"status": "healthy", "app_name": "OmniFlow", "version": "...", "environment": "..."}` with HTTP 200, no authentication required — this is the path
configured in `render.yaml`.

---

## Known limitations

- **Not yet live-deployed.** The Docker image, `render.yaml`, and single-origin
  frontend-serving wiring have been validated locally (all four `test_deployment.py`
  cases pass, the frontend builds and lints cleanly, and a local `docker build`
  produced a working 3.2GB image after the CPU-only-torch fix) — but the service has
  not yet been pushed to a live Render URL, and the five official assignment
  scenarios have not yet been run end-to-end against a real Gemini key in that
  environment. See [Deployment](#deployment-docker--render).
- **Render free-tier memory risk on the RAG path.** Loading
  `sentence-transformers`/PyTorch on the first request that actually triggers
  `rag_search` was observed locally to use roughly 600MB–1GB resident memory, close to
  or over Render's free-tier 512MB ceiling — this could cause an OOM restart
  specifically on that path. Direct-context answers and audio transcription (`tiny`
  Whisper) stay well under the ceiling in the same environment. This is a
  hosting-tier capacity constraint, not an application defect; addressing it would
  mean upgrading the Render plan or swapping to a smaller embedding model, both out
  of scope here. **This should be verified for real once deployed, before relying on
  it for the demo.**
- **The mandatory summarization triple-format (1-line + 3 bullets + 5-sentence) is
  not yet structurally enforced** — see [Not yet done](#not-yet-done) above.
- **`detected_urls` is unused schema** — YouTube/URL detection inside document
  content currently relies entirely on the intent-understanding LLM noticing it in a
  bounded content preview, with no deterministic regex fallback.
- **No multi-turn server-side state.** Every `/query` call is a fresh `AgentState`;
  clarification follow-ups are stitched together client-side (see
  `frontend/src/lib/clarification.ts`), not resumed server-side.
- **No persistence.** The FAISS index and chunk registry are in-memory only, rebuilt
  per request, and lost on process restart — by design for this project's scope.
- **Backend lint has no pinned ruleset** — see the note under
  [Run the test suite / Lint](#6-lint) above.

---

## Configuration reference

See `.env.example` for the complete, commented list. Highlights:

```env
APP_ENV=development
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.5-flash

OCR_LANGUAGE=eng
PDF_NATIVE_TEXT_CHAR_THRESHOLD=30

WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=4
RAG_SIMILARITY_THRESHOLD=0.2

MAX_AGENT_STEPS=6
MAX_TOOL_CALLS=6
MAX_RETRIES=2
```