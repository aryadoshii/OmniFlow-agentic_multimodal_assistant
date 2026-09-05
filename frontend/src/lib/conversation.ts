import type { ApiError } from '../api/client'
import type { OmniFlowResponse } from '../api/types'

/**
 * One request/response cycle in the UI's on-page history.
 *
 * This is NOT a backend conversation/session object -- OmniFlow has no
 * server-side session store (see lib/clarification.ts's docstring). It
 * exists purely so the page can keep showing past turns instead of
 * discarding them the moment a new one starts (requirement: "preserve the
 * current interaction/result in the UI").
 */
export interface ConversationTurn {
  id: string
  /** What the user actually typed, for display. */
  displayQuery: string
  /** What was actually sent to the backend (may fold in prior clarification context). */
  sentQuery: string
  isClarificationFollowUp: boolean
  fileNames: string[]
  status: 'loading' | 'success' | 'error'
  response: OmniFlowResponse | null
  error: ApiError | null
}
