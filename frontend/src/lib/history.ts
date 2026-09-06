import type { ExecutionTrace, NormalizedDocument, OmniFlowResponse } from '../api/types'
import type { ConversationDetail, ConversationSummary, StoredAssistantContent } from '../api/historyTypes'
import type { ConversationTurn } from './conversation'

/** Derives a short, human-readable conversation title from the first query. */
export function deriveConversationTitle(query: string): string {
  const trimmed = query.trim().replace(/\s+/g, ' ')
  if (trimmed.length <= 60) return trimmed || 'New conversation'
  return `${trimmed.slice(0, 57)}…`
}

export type HistoryGroupLabel = 'Today' | 'Yesterday' | 'Previous 7 days' | 'Older'

export interface HistoryGroup {
  label: HistoryGroupLabel
  items: ConversationSummary[]
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()
}

/** Groups conversations into Today / Yesterday / Previous 7 days / Older, by updated_at. */
export function groupConversationsByRecency(conversations: ConversationSummary[]): HistoryGroup[] {
  const now = startOfDay(new Date())
  const oneDay = 24 * 60 * 60 * 1000

  const buckets: Record<HistoryGroupLabel, ConversationSummary[]> = {
    Today: [],
    Yesterday: [],
    'Previous 7 days': [],
    Older: [],
  }

  for (const conversation of conversations) {
    const updated = startOfDay(new Date(conversation.updated_at))
    const diffDays = Math.round((now - updated) / oneDay)
    if (diffDays <= 0) buckets.Today.push(conversation)
    else if (diffDays === 1) buckets.Yesterday.push(conversation)
    else if (diffDays <= 7) buckets['Previous 7 days'].push(conversation)
    else buckets.Older.push(conversation)
  }

  return (['Today', 'Yesterday', 'Previous 7 days', 'Older'] as HistoryGroupLabel[])
    .map((label) => ({ label, items: buckets[label] }))
    .filter((group) => group.items.length > 0)
}

/** Reconstructs the ConversationTurn[] the UI renders from a restored ConversationDetail. */
export function conversationDetailToTurns(detail: ConversationDetail): ConversationTurn[] {
  const turns: ConversationTurn[] = []
  const attachmentNames = detail.attachments.map((a) => a.filename)

  for (let i = 0; i < detail.messages.length; i += 2) {
    const userMessage = detail.messages[i]
    const assistantMessage = detail.messages[i + 1]
    if (!userMessage || userMessage.role !== 'user') continue

    let response: OmniFlowResponse | null = null
    if (assistantMessage && assistantMessage.role === 'assistant') {
      try {
        const parsed = JSON.parse(assistantMessage.content) as StoredAssistantContent
        response = {
          session_id: null,
          status: parsed.status as OmniFlowResponse['status'],
          answer: parsed.answer,
          clarification_needed: false,
          clarification_prompt: null,
          normalized_documents: (parsed.normalized_documents ?? []) as NormalizedDocument[],
          execution_trace: (parsed.execution_trace ?? null) as ExecutionTrace | null,
          warnings: parsed.warnings ?? [],
          errors: parsed.errors ?? [],
        }
      } catch {
        response = null
      }
    }

    turns.push({
      id: userMessage.id,
      displayQuery: userMessage.content,
      sentQuery: userMessage.content,
      isClarificationFollowUp: false,
      // Attachment metadata is conversation-wide in this schema (see
      // omniflow/models/history.py), not per-turn -- attributing every
      // stored filename to only the first turn keeps a restored
      // conversation from claiming the same file was attached to every
      // message.
      fileNames: i === 0 ? attachmentNames : [],
      status: response ? 'success' : 'error',
      response,
      error: null,
    })
  }

  return turns
}
