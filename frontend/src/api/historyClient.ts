import type { ConversationDetail, ConversationSummary } from './historyTypes'
import { ApiError } from './client'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/+$/, '')

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
      ...init,
    })
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'The request could not be completed.',
      'NETWORK_ERROR',
      0,
      {},
    )
  }

  if (!response.ok) {
    let payload: { error?: { code: string; message: string; details: Record<string, unknown> } } | null = null
    try {
      payload = await response.json()
    } catch {
      // fall through
    }
    if (payload?.error) {
      throw new ApiError(payload.error.message, payload.error.code, response.status, payload.error.details ?? {})
    }
    throw new ApiError(`Request failed with status ${response.status}.`, 'UNKNOWN_ERROR', response.status, {})
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export interface SaveTurnAttachment {
  filename: string
  mime_type: string
}

export interface SaveTurnParams {
  query: string
  status: string
  answer: string | null
  warnings: string[]
  errors: string[]
  execution_trace: unknown
  normalized_documents: unknown[]
  attachments: SaveTurnAttachment[]
}

/** POST /conversations -- creates a new, empty conversation. */
export async function createConversation(title: string): Promise<ConversationSummary> {
  return requestJson<ConversationSummary>('/conversations', {
    method: 'POST',
    body: JSON.stringify({ title }),
  })
}

/** GET /conversations -- lists all conversations, most recently updated first. */
export async function listConversations(): Promise<ConversationSummary[]> {
  return requestJson<ConversationSummary[]>('/conversations')
}

/** GET /conversations/{id} -- fetches one conversation's full messages/attachments. */
export async function getConversation(conversationId: string): Promise<ConversationDetail> {
  return requestJson<ConversationDetail>(`/conversations/${encodeURIComponent(conversationId)}`)
}

/** POST /conversations/{id}/turns -- persists one query + response exchange. */
export async function saveTurn(conversationId: string, params: SaveTurnParams): Promise<ConversationSummary> {
  return requestJson<ConversationSummary>(`/conversations/${encodeURIComponent(conversationId)}/turns`, {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

/** DELETE /conversations/{id} -- permanently deletes a conversation. */
export async function deleteConversation(conversationId: string): Promise<void> {
  await requestJson<void>(`/conversations/${encodeURIComponent(conversationId)}`, { method: 'DELETE' })
}
