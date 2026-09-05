import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ApiError } from '../api/client'
import type { OmniFlowResponse } from '../api/types'
import { describeApiError } from '../lib/errorDisplay'
import { describeTraceStep } from '../lib/traceDisplay'
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
        <p>Running the agent…</p>
      </div>
    )
  }

  if (status === 'error' && error) {
    const { title, message } = describeApiError(error)
    return (
      <div className="response-area response-area--error" role="alert">
        <p className="response-area__error-title">{title}</p>
        <p>{message}</p>
      </div>
    )
  }

  if (status === 'success' && response) {
    return <SuccessResponse data={response} />
  }

  return null
}

function SuccessResponse({ data }: { data: OmniFlowResponse }) {
  const isAwaitingClarification = data.status === 'awaiting_clarification'

  return (
    <div className="response-area response-area--success">
      <div className="response-area__meta">
        <StatusBadge status={data.status} />
        {data.execution_trace?.total_duration_ms != null && (
          <span className="response-area__duration">
            {data.execution_trace.total_duration_ms.toFixed(0)} ms
          </span>
        )}
      </div>

      {data.clarification_needed && data.clarification_prompt && (
        <div className="response-area__clarification" role="status">
          <p className="response-area__clarification-title">Clarification needed</p>
          <p>{data.clarification_prompt}</p>
          <p className="response-area__clarification-hint">Answer below to continue.</p>
        </div>
      )}

      {data.answer && (
        <div className="response-area__answer">
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

      {data.errors.length > 0 && (
        <details className="response-area__section" open>
          <summary>Errors ({data.errors.length})</summary>
          <ul>
            {data.errors.map((message, index) => (
              <li key={index}>{message}</li>
            ))}
          </ul>
        </details>
      )}

      {data.warnings.length > 0 && (
        <details className="response-area__section">
          <summary>Warnings ({data.warnings.length})</summary>
          <ul>
            {data.warnings.map((message, index) => (
              <li key={index}>{message}</li>
            ))}
          </ul>
        </details>
      )}

      {data.normalized_documents.length > 0 && (
        <details className="response-area__section">
          <summary>Processed documents ({data.normalized_documents.length})</summary>
          <ul>
            {data.normalized_documents.map((doc) => (
              <li key={doc.id}>
                <strong>{doc.filename}</strong> — {doc.source_type} ({doc.extraction_method})
                {doc.warnings.length > 0 && (
                  <span className="response-area__doc-warning"> · {doc.warnings.join('; ')}</span>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}

      {data.execution_trace && data.execution_trace.steps.length > 0 && (
        <details className="response-area__section">
          <summary>How the agent got here ({data.execution_trace.steps.length} steps)</summary>
          <ol>
            {data.execution_trace.steps.map((step, index) => {
              const { label, group } = describeTraceStep(step)
              return (
                <li
                  key={index}
                  className={`response-area__trace-step response-area__trace-step--${step.status} response-area__trace-step--${group}`}
                >
                  {label} — {step.status}
                  {step.duration_ms != null ? ` · ${step.duration_ms.toFixed(1)} ms` : ''}
                  {step.error_message ? `: ${step.error_message}` : ''}
                </li>
              )
            })}
          </ol>
        </details>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`response-area__status-badge response-area__status-badge--${status}`}>
      {status.replace(/_/g, ' ')}
    </span>
  )
}
