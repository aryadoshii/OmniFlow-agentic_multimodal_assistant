import { useState } from 'react'
import { AppShell } from './components/AppShell'
import { ConversationTurnCard } from './components/ConversationTurnCard'
import { EmptyState } from './components/EmptyState'
import { FileUploadArea } from './components/FileUploadArea'
import { QueryInput } from './components/QueryInput'
import { UploadedFileList } from './components/UploadedFileList'
import { ApiError, queryAgent } from './api/client'
import { buildClarificationFollowUp } from './lib/clarification'
import type { ConversationTurn } from './lib/conversation'
import { findRejectedFilename } from './lib/errorDisplay'
import { createStagedFile } from './lib/files'
import type { StagedFile } from './lib/files'
import { generateId } from './lib/id'
import './App.css'

function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  // client.ts already wraps network/timeout/HTTP failures as ApiError --
  // this only guards against a genuinely unexpected bug elsewhere.
  return new ApiError(error instanceof Error ? error.message : 'An unexpected error occurred.', 'UNKNOWN_ERROR', 0, {})
}

interface PendingClarification {
  originalQuery: string
  question: string | null
}

function App() {
  // A random id generated once per page load, sent as `session_id` and
  // echoed back by the backend. NOTE: the backend does not maintain any
  // server-side session store keyed by this -- see lib/clarification.ts --
  // so this is a passthrough correlation label only, not proof of shared
  // conversation state.
  const [sessionId] = useState(() => generateId())
  const [files, setFiles] = useState<StagedFile[]>([])
  const [turns, setTurns] = useState<ConversationTurn[]>([])
  // Tracks a still-unanswered clarification question separately from
  // `turns`. This must survive a FAILED answer attempt: if answering a
  // clarification hits a network error or timeout, the turn's status
  // becomes 'error', but the clarification itself was never actually
  // resolved -- deriving "am I mid-clarification" from only the last
  // turn's status would silently drop back to fresh-query mode on retry,
  // losing the original question and context.
  const [pendingClarification, setPendingClarification] = useState<PendingClarification | null>(null)

  const lastTurn = turns.length > 0 ? turns[turns.length - 1] : null
  const isSubmitting = lastTurn?.status === 'loading'
  const isAwaitingClarification = pendingClarification !== null

  function handleFilesSelected(newFiles: File[]) {
    setFiles((prev) => [...prev, ...newFiles.map(createStagedFile)])
  }

  function handleRemoveFile(id: string) {
    setFiles((prev) => prev.filter((staged) => staged.id !== id))
  }

  async function handleSubmit(text: string) {
    // Belt-and-suspenders: QueryInput/FileUploadArea are already disabled
    // while a request is in flight, but never start a second request even
    // if this were somehow called again.
    if (isSubmitting) return

    const sentQuery = pendingClarification
      ? buildClarificationFollowUp(pendingClarification.originalQuery, pendingClarification.question, text)
      : text

    const turnId = generateId()
    const fileNames = files.map((staged) => staged.file.name)

    setTurns((prev) => [
      ...prev,
      {
        id: turnId,
        displayQuery: text,
        sentQuery,
        isClarificationFollowUp: isAwaitingClarification,
        fileNames,
        status: 'loading',
        response: null,
        error: null,
      },
    ])

    try {
      const response = await queryAgent({
        query: sentQuery,
        sessionId,
        files: files.map((staged) => staged.file),
      })
      setTurns((prev) => prev.map((turn) => (turn.id === turnId ? { ...turn, status: 'success', response } : turn)))
      // Resolved (a real answer came back) unless the backend is asking
      // ANOTHER clarifying question -- in which case that becomes the new
      // pending one, still anchored to the same original request.
      setPendingClarification(
        response.status === 'awaiting_clarification'
          ? { originalQuery: sentQuery, question: response.clarification_prompt }
          : null,
      )
    } catch (caught) {
      const error = toApiError(caught)
      setTurns((prev) => prev.map((turn) => (turn.id === turnId ? { ...turn, status: 'error', error } : turn)))
      // Deliberately NOT touching pendingClarification here: a failed
      // request (network error, timeout, backend failure) never actually
      // resolved the clarification, so it must still be pending for the
      // user's next attempt.

      const rejectedFilename = findRejectedFilename(error)
      if (rejectedFilename) {
        setFiles((prev) =>
          prev.map((staged) => (staged.file.name === rejectedFilename ? { ...staged, issue: error.message } : staged)),
        )
      }
    }
  }

  return (
    <AppShell>
      <section className="app__panel" aria-label="Query and file input">
        <FileUploadArea onFilesSelected={handleFilesSelected} disabled={isSubmitting} />
        <UploadedFileList files={files} onRemove={handleRemoveFile} disabled={isSubmitting} />
        <QueryInput
          onSubmit={handleSubmit}
          disabled={isSubmitting}
          mode={isAwaitingClarification ? 'clarify' : 'ask'}
          clarificationQuestion={pendingClarification?.question}
        />
      </section>
      <section className="app__panel app__conversation" aria-label="Agent responses" aria-live="polite">
        {turns.length === 0 ? (
          <EmptyState />
        ) : (
          [...turns]
            .reverse()
            .map((turn, index) => <ConversationTurnCard key={turn.id} turn={turn} isLatest={index === 0} />)
        )}
      </section>
    </AppShell>
  )
}

export default App
