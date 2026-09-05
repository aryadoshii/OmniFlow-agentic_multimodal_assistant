import type { ApiErrorPayload, IngestionResponse, OmniFlowResponse } from './types'

// Unset in development: same-origin "/api/*" is proxied to the local
// backend by vite.config.ts. Set VITE_API_BASE_URL at build time to point
// at a real backend origin -- never hardcoded here.
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/+$/, '')

// The browser's own `fetch` has no timeout at all -- without one, a hung
// backend/model call leaves the UI stuck on "Working…" forever with no
// recovery path but a page reload. The agent loop is bounded server-side
// (max_agent_steps/max_tool_calls, see omniflow/config.py) but a multi-step
// run with several LLM calls can legitimately take a while, so this is
// generous, not a request-latency budget.
const REQUEST_TIMEOUT_MS = 120_000

/**
 * Thrown for any non-2xx API response. Carries the backend's own error
 * envelope (code/message/details from error_handlers.py) so callers can
 * distinguish e.g. an UNSUPPORTED_FILE_TYPE from a CONFIGURATION_ERROR
 * without parsing the message string.
 */
export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly details: Record<string, unknown>

  constructor(message: string, code: string, status: number, details: Record<string, unknown>) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.details = details
  }
}

async function throwForErrorResponse(response: Response): Promise<never> {
  let payload: ApiErrorPayload | null = null
  try {
    payload = (await response.json()) as ApiErrorPayload
  } catch {
    // Response body wasn't JSON (e.g. a proxy/network-level failure page) --
    // fall through to the generic error below.
  }

  if (payload?.error) {
    throw new ApiError(payload.error.message, payload.error.code, response.status, payload.error.details ?? {})
  }
  throw new ApiError(`Request failed with status ${response.status}.`, 'UNKNOWN_ERROR', response.status, {})
}

async function postForm<T>(path: string, formData: FormData): Promise<T> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(
        `The request took longer than ${REQUEST_TIMEOUT_MS / 1000} seconds and was cancelled.`,
        'TIMEOUT',
        0,
        {},
      )
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'The request could not be completed.',
      'NETWORK_ERROR',
      0,
      {},
    )
  } finally {
    clearTimeout(timeoutId)
  }

  if (!response.ok) {
    await throwForErrorResponse(response)
  }

  return (await response.json()) as T
}

export interface QueryAgentParams {
  /** The user's question or instruction. Required, non-blank (enforced by the backend). */
  query: string
  /** Optional session/conversation identifier, echoed back by the backend. */
  sessionId?: string
  /** Optional file uploads (PDF, image, audio, text) -- see IngestParams for accepted formats. */
  files?: File[]
}

/** POST /query -- runs the full agent workflow. See omniflow/api/routes/agent.py. */
export async function queryAgent({ query, sessionId, files }: QueryAgentParams): Promise<OmniFlowResponse> {
  const formData = new FormData()
  formData.append('query', query)
  if (sessionId) {
    formData.append('session_id', sessionId)
  }
  for (const file of files ?? []) {
    formData.append('files', file)
  }
  return postForm<OmniFlowResponse>('/query', formData)
}

export interface IngestParams {
  /** Optional plain text input. At least one of text/files is required by the backend. */
  text?: string
  /** Optional file uploads (PDF, image, audio, text). */
  files?: File[]
}

/** POST /ingest -- lower-level ingestion-only endpoint. See omniflow/api/routes/ingest.py. */
export async function ingestFiles({ text, files }: IngestParams): Promise<IngestionResponse> {
  const formData = new FormData()
  if (text) {
    formData.append('text', text)
  }
  for (const file of files ?? []) {
    formData.append('files', file)
  }
  return postForm<IngestionResponse>('/ingest', formData)
}
