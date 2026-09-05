# OmniFlow Frontend

A single-page React + Vite + TypeScript client for the OmniFlow FastAPI backend.

## What this is

- One page: a query box, a file drop area, and a response panel.
- Talks to the backend's existing `/query` and `/ingest` endpoints only — no
  ingestion, RAG, planning, tool execution, or LLM logic is reimplemented
  here. This app renders what the backend returns.

## Development

The backend must be running separately (from the repo root):

```bash
uvicorn omniflow.main:app --reload
```

Then, in this directory:

```bash
npm install
npm run dev
```

Requests to `/api/*` are proxied to `http://127.0.0.1:8000` by
`vite.config.ts` (matching `omniflow.config.Settings`' default HOST/PORT),
with the `/api` prefix stripped before forwarding — the backend's own
routes are mounted at root (`/query`, `/ingest`, `/health`). Set
`OMNIFLOW_BACKEND_ORIGIN` before running `npm run dev` if your backend runs
elsewhere.

## Configuration

No production URL is hardcoded. See `.env.example`: set
`VITE_API_BASE_URL` at build time to point the built app at a real backend
origin. Left unset, the app calls same-origin `/api/*` (only meaningful
behind the dev proxy above, or an equivalent reverse proxy in production).

## Checks

```bash
npm run build   # tsc --noEmit-equivalent project build + vite build
npm run lint    # oxlint
```
