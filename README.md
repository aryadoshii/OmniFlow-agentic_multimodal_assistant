# OmniFlow: Agentic Multimodal AI Assistant

OmniFlow is an agentic multimodal assistant engineered for intelligent ingestion, autonomous planning, deterministic tool execution, conditional RAG, and multi-input synthesis.

> **Current Status: Phase 2 Implemented (Multimodal Ingestion Layer)**  
> This repository currently contains the Phase 1 architectural foundation and Phase 2 multimodal ingestion layer. Downstream agent tools (YouTube transcripts, FAISS RAG), LangGraph orchestration, LLM reasoning, and UI components are scheduled for subsequent phases.

---

## Status Summary

### Implemented
- **Phase 1: Architecture & Foundation**: Modular FastAPI structure, domain models (`NormalizedDocument`, `AgentState`), centralized configuration, exception hierarchy, error handlers, and logging.
- **Phase 2: Multimodal Ingestion Layer**: Deterministic processors converting plain text, PDFs (native text with per-page OCR fallback), images (JPG/PNG via Tesseract OCR), and audio (WAV/MP3/M4A via faster-whisper) into unified `NormalizedDocument` representations.

### Not Yet Implemented (Scheduled for Phases 3-5)
- YouTube URL detection & transcript retrieval
- FAISS local vector retrieval & conditional RAG
- LangGraph orchestration, state machine & planner
- Ambiguity clarification flow
- Gemini LLM reasoning & synthesis
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

Available Phase 2 settings:
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

### 7. Run the Test Suite

```bash
pytest -v
```
