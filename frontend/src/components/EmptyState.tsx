import './EmptyState.css'

const CAPABILITIES = [
  'Summarize or extract action items from a PDF or text document',
  'Transcribe and summarize an audio recording, with its duration',
  'Explain a code screenshot — language, what it does, and any bugs',
  'Follow a YouTube link mentioned inside a document and summarize it',
  'Compare multiple documents (e.g. an audio recording and a PDF)',
  'Answer questions grounded in whatever you attach — and say so plainly when it can\'t find an answer',
]

/** Shown before the first query -- explains what OmniFlow can actually do. */
export function EmptyState() {
  return (
    <div className="empty-state">
      <p className="empty-state__lead">
        Ask a question, optionally with files attached (PDF, image, audio, or text).
      </p>
      <ul className="empty-state__list">
        {CAPABILITIES.map((capability) => (
          <li key={capability}>{capability}</li>
        ))}
      </ul>
    </div>
  )
}
