import { useState } from 'react'
import { ChevronDown, CircleAlert, CircleCheck, TriangleAlert } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ApiError } from '../api/client'
import type { OmniFlowResponse } from '../api/types'
import { describeApiError } from '../lib/errorDisplay'
import { AgentTrace } from './AgentTrace'
import { CrossSourceAnalysisView } from './CrossSourceAnalysisView'
import { EvidenceSection } from './EvidenceSection'
import { ProcessedDocuments } from './ProcessedDocuments'
import './ResponseArea.css'

interface ResponseAreaProps {
  status: 'loading' | 'success' | 'error'
  response: OmniFlowResponse | null
  error: ApiError | null
}

/** Renders the loading/error/success outcome of one /query call. */
export function ResponseArea({ status, response, error }: ResponseAreaProps) {
  if (status === 'loading') {
    return (
      <div className="response-area response-area--loading" role="status" aria-live="polite">
        <span className="response-area__spinner" aria-hidden="true" />
        <p>OmniFlow is thinking…</p>
      </div>
    )
  }

  if (status === 'error' && error) {
    const { title, message } = describeApiError(error)
    return (
      <div className="response-area response-area--top-error" role="alert">
        <p className="response-area__error-title">
          <CircleAlert size={16} /> {title}
        </p>
        <p className="response-area__error-message">{message}</p>
        <ErrorDetails code={error.code} details={error.details} />
      </div>
    )
  }

  if (status === 'success' && response) {
    return <SuccessResponse data={response} />
  }

  return null
}

function ErrorDetails({ code, details }: { code: string; details: Record<string, unknown> }) {
  const [isOpen, setIsOpen] = useState(false)
  const hasDetails = Object.keys(details).length > 0
  if (!hasDetails) return null

  return (
    <div className="response-area__error-details">
      <button type="button" onClick={() => setIsOpen((open) => !open)} aria-expanded={isOpen}>
        Technical details ({code})
        <ChevronDown size={13} className={isOpen ? 'response-area__chevron--open' : ''} />
      </button>
      {isOpen && <pre>{JSON.stringify(details, null, 2)}</pre>}
    </div>
  )
}

function SuccessResponse({ data }: { data: OmniFlowResponse }) {
  const isAwaitingClarification = data.status === 'awaiting_clarification'
  const isFailed = data.status === 'failed'

  return (
    <div className="response-area response-area--success">
      <div className="response-area__meta">
        <span className="response-area__brand">OmniFlow</span>
        <StatusBadge status={data.status} />
        {data.execution_trace?.total_duration_ms != null && (
          <span className="response-area__duration">
            {(data.execution_trace.total_duration_ms / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      {data.clarification_needed && data.clarification_prompt && (
        <div className="response-area__clarification" role="status">
          <p className="response-area__clarification-title">
            <TriangleAlert size={14} /> Clarification needed
          </p>
          <p>{data.clarification_prompt}</p>
          <p className="response-area__clarification-hint">Answer below to continue.</p>
        </div>
      )}

      {data.answer && (
        <div className="response-area__answer markdown">
          {/* GFM (remark-gfm) enables tables/strikethrough/autolinks --
              comparison-style answers (e.g. "compare this audio with the
              PDF") commonly come back as a markdown table, which would
              otherwise render as broken plain text with literal "|" characters. */}
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.answer}</ReactMarkdown>
        </div>
      )}

      {!data.answer && !isAwaitingClarification && (
        <p className="response-area__answer response-area__answer--empty">No answer was produced.</p>
      )}

      {data.cross_source_analysis && <CrossSourceAnalysisView analysis={data.cross_source_analysis} />}

      {data.errors.length > 0 && (
        <CollapsibleSection
          title={`Errors (${data.errors.length})`}
          tone="error"
          defaultOpen={isFailed}
          items={data.errors}
        />
      )}

      {data.warnings.length > 0 && (
        <CollapsibleSection title={`Warnings (${data.warnings.length})`} tone="warning" items={data.warnings} />
      )}

      <ProcessedDocuments documents={data.normalized_documents} />

      <EvidenceSection evidence={data.evidence} />

      {data.execution_trace && <AgentTrace trace={data.execution_trace} />}
    </div>
  )
}

function CollapsibleSection({
  title,
  tone,
  items,
  defaultOpen = false,
}: {
  title: string
  tone: 'error' | 'warning'
  items: string[]
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  return (
    <div className={`response-area__section response-area__section--${tone}`}>
      <button type="button" onClick={() => setIsOpen((open) => !open)} aria-expanded={isOpen}>
        {tone === 'error' ? <CircleAlert size={14} /> : <TriangleAlert size={14} />}
        {title}
        <ChevronDown size={14} className={isOpen ? 'response-area__chevron--open' : ''} />
      </button>
      {isOpen && (
        <ul>
          {items.map((message, index) => (
            <li key={index}>{message}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

const STATUS_CONFIG: Record<string, { label: string; icon: typeof CircleCheck }> = {
  completed: { label: 'Completed', icon: CircleCheck },
  failed: { label: 'Failed', icon: CircleAlert },
  awaiting_clarification: { label: 'Needs input', icon: TriangleAlert },
}

function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIG[status] ?? { label: status.replace(/_/g, ' '), icon: CircleCheck }
  const Icon = config.icon
  return (
    <span className={`response-area__status-badge response-area__status-badge--${status}`}>
      <Icon size={13} strokeWidth={2.5} />
      {config.label}
    </span>
  )
}
