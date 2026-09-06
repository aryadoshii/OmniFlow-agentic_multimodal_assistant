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
# from the SAME origin in production (see backend/main.py), so API calls
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

# uv itself -- copied as a static, self-contained binary from Astral's
# official distroless image (pinned to the version this project's uv.lock
# was generated with). No pip-install-uv step, no extra Python packages
# pulled into this stage just to install our real ones.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/

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

# Created before installing anything, and switched to BEFORE `uv sync` /
# copying application code, so every file the rest of this stage creates
# (the venv, backend/, frontend/dist, database/) is appuser-owned from the
# moment it's written. A single trailing `chown -R /app` after gigabytes of
# venv/dependencies already exist would make Docker's overlay filesystem
# copy up the ENTIRE tree into a new layer just to change ownership
# metadata -- observed to roughly double this image's size (measured via
# `docker history`) for no functional benefit.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
RUN chown appuser:appuser /app

# UV_LINK_MODE=copy avoids hardlink warnings when uv's cache and the venv
# live on different layers/filesystems. UV_PYTHON_DOWNLOADS=never keeps uv
# from ever reaching out for its own interpreter -- it must use the
# python:3.12-slim interpreter already on PATH in this image.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --chown=appuser:appuser pyproject.toml uv.lock ./

USER appuser

# Install Python dependencies before copying application code so this layer
# is cached unless pyproject.toml/uv.lock change. --frozen refuses to
# update the lockfile inside the build -- the committed uv.lock is the
# single source of truth for exactly what gets installed, matching what
# `pytest`/`ruff` were already validated against locally. --no-dev skips
# the dev-only dependency group (pytest, ruff, httpx) -- tests run
# locally/in CI, never inside the deployed container. The cache mount keeps
# uv's downloaded/built wheel cache OUTSIDE this layer entirely (reused
# across builds for speed) instead of baking it into the image.
#
# pyproject.toml's [tool.uv.sources] pins torch to PyTorch's CPU-only wheel
# index. Without it, resolving sentence-transformers'/faster-whisper's
# torch dependency would pull the CUDA-enabled build (several GB of unused
# GPU libraries) even though the app only ever runs embeddings/Whisper on
# CPU (EMBEDDING_DEVICE and WHISPER_DEVICE both default to "cpu" -- see
# backend/config.py).
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

# Application code and the frontend build output (from stage 1), at the
# exact relative path backend/main.py expects (frontend/dist next to
# backend/, both under the repo root / WORKDIR). --chown here (not a later
# chown -R) is what keeps this a cheap, single-purpose layer.
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser --from=frontend-build /app/frontend/dist ./frontend/dist

# The SQLite history database's directory (see backend/services/
# history_store.py / HISTORY_DB_PATH) -- created here (already appuser-
# owned, since /app itself is) so it exists before the app ever tries to
# write to it; the app itself also creates it defensively at startup
# (init_db()) if it's ever missing.
#
# Uploaded files are written to the OS temp directory and always removed
# after processing (see backend/services/temp_manager.py) -- nothing here
# persists across container restarts, and no volume is declared, so
# uploads/temp files stay ephemeral by construction, not by extra
# configuration. The SQLite history file at database/ is the one
# exception: it persists only for the life of this container's filesystem,
# and is lost on redeploy/restart on platforms without a mounted
# persistent disk (e.g. Render's free tier).
RUN mkdir -p database

ENV APP_ENV=production \
    HOST=0.0.0.0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Informational only -- Render (and most PaaS targets) inject the real
# listen port via $PORT at runtime; the CMD below reads it dynamically.
# GEMINI_API_KEY and any other secret are supplied the same way (via the
# platform's environment variable configuration) and are never baked into
# this image or its build context (see .dockerignore).
EXPOSE 8000

# No --reload (production process, not a dev server). $PORT defaults to
# 8000 for a local `docker run` where the platform doesn't set it.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
