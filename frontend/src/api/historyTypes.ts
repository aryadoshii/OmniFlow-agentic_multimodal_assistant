/**
 * Types mirroring omniflow/models/history.py -- the SQLite-backed
 * conversation history API (sidebar). Kept separate from types.ts (the
 * /query and /ingest contracts) since this is a distinct, purely additive
 * API surface that never touches the agent workflow.
 */

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  attachment_count: number
}

export type MessageRole = 'user' | 'assistant'

export interface MessageOut {
  id: string
  role: MessageRole
  content: string
  created_at: string
}

export interface AttachmentOut {
  id: string
  filename: string
  mime_type: string
  created_at: string
}

export interface ConversationDetail {
  id: string
  title: string
  created_at: string
  updated_at: string
  messages: MessageOut[]
  attachments: AttachmentOut[]
}

/** Parsed shape of an assistant message's JSON-encoded `content` field. */
export interface StoredAssistantContent {
  status: string
  answer: string | null
  warnings: string[]
  errors: string[]
  execution_trace: unknown
  normalized_documents: unknown[]
}
