# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build the frontend (Vite + React + TypeScript) into static assets.
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend

# Install dependencies first so this layer is cached unless package*.json changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Single-service deployment: the built SPA and the FastAPI API are served
# from the SAME origin in production (see omniflow/main.py), so API calls
# must hit the backend's real paths directly (e.g. "/query") instead of the
# dev-only "/api" prefix Vite's local dev proxy strips (vite.config.ts).
# Setting this to an empty string makes src/api/client.ts's existing
# `VITE_API_BASE_URL ?? '/api'` fallback resolve to same-origin, prefix-less
# paths -- no source change needed, this is purely a build-time setting.
ENV VITE_API_BASE_URL=""
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: the FastAPI backend runtime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS backend

# System dependencies required by the existing (unchanged) Python
# dependencies -- not new capabilities:
#   - tesseract-ocr: pytesseract (image/scanned-PDF OCR)
#   - ffmpeg:        faster-whisper's audio decoding (mp3/m4a/wav)
#   - libgomp1:      OpenMP runtime required by faiss-cpu/torch on Debian
#                     slim images (a well-known missing-.so failure without it)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying application code so this layer
# is cached unless requirements.txt changes. A generous timeout/retry count
# is needed because torch alone is a large download -- default pip settings
# can time out on it over a slow or congested connection.
#
# --extra-index-url points pip at PyTorch's CPU-only wheel index. Without
# this, pip's default resolution of sentence-transformers' torch dependency
# pulls in the CUDA-enabled build (several GB of unused GPU libraries) even
# though the app only ever runs embeddings/Whisper on CPU (EMBEDDING_DEVICE
# and WHISPER_DEVICE both default to "cpu" -- see omniflow/config.py). This
# is the same torch version pip would otherwise choose, just the CPU build
# of it -- not a dependency change.
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 --retries=10 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Application code and the frontend build output (from stage 1), at the
# exact relative path omniflow/main.py expects (frontend/dist next to
# omniflow/, both under the repo root / WORKDIR).
COPY omniflow ./omniflow
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Run as a non-root user. Uploaded files are written to the OS temp
# directory and always removed after processing (see
# omniflow/services/temp_manager.py) -- nothing here persists across
# container restarts, and no volume is declared, so uploads/temp files stay
# ephemeral by construction, not by extra configuration.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV APP_ENV=production \
    HOST=0.0.0.0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Informational only -- Render (and most PaaS targets) inject the real
# listen port via $PORT at runtime; the CMD below reads it dynamically.
# GEMINI_API_KEY and any other secret are supplied the same way (via the
# platform's environment variable configuration) and are never baked into
# this image or its build context (see .dockerignore).
EXPOSE 8000

# No --reload (production process, not a dev server). $PORT defaults to
# 8000 for a local `docker run` where the platform doesn't set it.
CMD ["sh", "-c", "exec uvicorn omniflow.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
